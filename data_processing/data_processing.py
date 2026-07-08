# -*- coding:utf-8 -*-

# Importation des modules
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from utils import filter_SN # type:ignore
from scipy.signal import medfilt, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Union, Optional
import pywt  # Pour wavelet transforms
from scipy.ndimage import grey_opening, white_tophat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

def load_data(
	filepath: str,
	mslevel: str = "1",
	sorting_by: list[str] = ["rt", "mz", "dt"]
) -> pd.DataFrame:
	"""
	Charge un fichier Parquet de données UPLC-HRMS-IMS et prépare les colonnes essentielles.

	Cette fonction lit un fichier contenant des scans spectrométriques, filtre les données
	selon le niveau MS (par défaut : MS1), trie les lignes si nécessaire, puis convertit
	le tout en DataFrame pandas prêt à l'analyse.

	Args:
		filepath (str): Chemin vers le fichier Parquet.
		mslevel (str): Niveau de masse à conserver (ex: '1' pour MS1).
		sorting_by (list[str]): Liste de colonnes pour trier les données.

	Returns:
		pd.DataFrame: Données filtrées et formatées avec les colonnes : 'mz', 'rt', 'dt', 'intensity'.
	"""
	# === Étape 1 : Définition des colonnes à lire et des filtres de sélection ===
	columns = ["rt", "mz", "dt", "intensity"]
	filters = [("mslevel", "==", mslevel)]

	try:
		# === Étape 2 : Lecture du fichier Parquet avec les colonnes et filtres spécifiés ===
		data = pq.read_table(filepath, filters=filters)
	except FileNotFoundError:
		raise FileNotFoundError(f"Le fichier '{filepath}' est introuvable.")

	# === Étape 3 : Tri éventuel des données selon les colonnes choisies ===
	if sorting_by:
		sort_keys = [(col, "ascending") for col in sorting_by]
		sorted_indices = pc.sort_indices(data, sort_keys=sort_keys)
		data = data.take(sorted_indices)

	# === Étape 4 : Conversion vers pandas.DataFrame ===
	data: pd.DataFrame = data.to_pandas()

	# Conversion explicite des colonnes en float (sécurité typage)
	data = data.astype({col: float for col in columns})

	# Réduction aux colonnes essentielles et réinitialisation de l'index
	return data.reset_index(drop=True)


class DataProcessor:
	"""
	Classe utilitaire pour charger et manipuler un fichier Parquet de données spectrométriques.

	Attributs :
		original_data (pd.DataFrame) : Données d'origine chargées depuis le fichier.
		data (pd.DataFrame) : Copie modifiable utilisée pour les traitements.
	"""

	def __init__(
		self,
		filepath: str,
		mslevel: str = "1",
		sorting_by: list[str] = ["rt", "mz", "dt"]
	) -> "DataProcessor":
		"""
		Initialise l'instance en chargeant les données et en créant une copie de travail.

		Args:
			filepath (str): Chemin du fichier Parquet à charger.
			mslevel (str): Niveau MS à filtrer (par défaut : '1' pour MS1).
			sorting_by (list[str]): Liste des colonnes à utiliser pour trier les données.
		"""
		# Chargement des données d'origine
		self.original_data: pd.DataFrame = load_data(
			filepath=filepath,
			mslevel=mslevel,
			sorting_by=sorting_by
		)

		# Création d'une copie modifiable pour le traitement
		self.data: pd.DataFrame = self.original_data.copy()

	def reset_data(self) -> "DataProcessor":
		"""
		Réinitialise les données de travail à leur état d'origine.

		Returns:
			DataProcessor: L'instance elle-même (permet le chaînage des appels).
		"""
		self.data = self.original_data.copy()
		return self

	# ============================================================================
	# MÉTHODE ORIGINALE - BASELINE CORRECTION
	# ============================================================================

	def baseline_correction(self, window: int = 50) -> "DataProcessor":
		"""
		Applique une correction de la ligne de base sur la colonne 'intensity'.

		Cette méthode utilise un filtre médian pour estimer le fond de bruit local,
		qu'elle soustrait ensuite à la courbe d'intensité. Les valeurs négatives
		sont ensuite corrigées à zéro.

		Args:
			window (int): Taille de la fenêtre pour le filtre médian (doit être impaire).

		Returns:
			DataProcessor: L'instance courante, avec la correction appliquée.
		"""
		# S'assurer que la fenêtre est impaire (exigence du filtre médian)
		if window % 2 == 0:
			window += 1

		# Récupération des valeurs d'intensité
		intensity_values = self.data["intensity"].values

		# Estimation du fond par filtre médian (scipy.signal.medfilt)
		baseline = medfilt(intensity_values, kernel_size=window)

		# Soustraction du fond et suppression des valeurs négatives
		corrected_intensity = intensity_values - baseline
		corrected_intensity = np.clip(corrected_intensity, a_min=0, a_max=None)

		# Mise à jour des données avec les intensités corrigées
		self.data["intensity"] = corrected_intensity

		# Retour de l'instance (permet le chaînage des méthodes)
		return self
	

	# ============================================================================
	# NOUVELLES MÉTHODES - BASELINE CORRECTION AVANCÉE
	# ============================================================================

	def baseline_als(
		self, 
		lam: float = 1e6, 
		p: float = 0.01, 
		niter: int = 10
	) -> "DataProcessor":
		"""
		Correction de baseline par Asymmetric Least Squares (ALS).
		
		Méthode de référence pour la spectrométrie, très efficace pour séparer
		le signal des pics de la baseline qui dérive lentement.
		
		Référence: P. Eilers & H. Boelens (2005)
		"Baseline Correction with Asymmetric Least Squares Smoothing"
		
		Args:
			lam (float): Paramètre de lissage (10^2 à 10^9, défaut 10^6)
			            Plus grand = baseline plus lisse
			p (float): Paramètre d'asymétrie (0.001 à 0.1, défaut 0.01)
			          Plus petit = baseline passe sous les pics
			niter (int): Nombre d'itérations (5-20)
		
		Returns:
			DataProcessor: Instance avec baseline corrigée
		"""
		y = self.data["intensity"].values
		L = len(y)
		
		# Matrice de différences de second ordre (pour le lissage)
		D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
		D = lam * D.dot(D.transpose())
		
		# Initialisation des poids
		w = np.ones(L)
		W = sparse.spdiags(w, 0, L, L)
		
		# Itérations ALS
		for i in range(niter):
			W.setdiag(w)
			Z = W + D
			z = spsolve(Z, w * y)
			w = p * (y > z) + (1 - p) * (y < z)
		
		# Soustraction de la baseline
		baseline = z
		corrected = y - baseline
		corrected = np.clip(corrected, a_min=0, a_max=None)
		
		self.data["intensity"] = corrected
		return self

	def baseline_tophat(
		self, 
		structure_size: int = 50
	) -> "DataProcessor":
		"""
		Correction de baseline par morphologie mathématique (Top-Hat).
		
		Utilise l'opération top-hat (chapeau haut-de-forme) qui extrait
		les pics en soustrayant l'ouverture morphologique.
		Très rapide et efficace pour des pics bien définis.
		
		Args:
			structure_size (int): Taille de l'élément structurant
			                     Doit être plus large que les pics
		
		Returns:
			DataProcessor: Instance avec baseline corrigée
		"""
		y = self.data["intensity"].values
		
		# Élément structurant (ligne 1D)
		structure = np.ones(structure_size)
		
		# Opération white top-hat
		corrected = white_tophat(y, structure=structure)
		corrected = np.clip(corrected, a_min=0, a_max=None)
		
		self.data["intensity"] = corrected
		return self

	def baseline_rolling_ball(
		self, 
		window: int = 100, 
		percentile: float = 5.0
	) -> "DataProcessor":
		"""
		Correction de baseline par algorithme Rolling Ball.
		
		Simule une balle qui roule sous le signal, créant une baseline
		qui suit les variations lentes sans toucher les pics.
		
		Args:
			window (int): Taille de la fenêtre mobile (rayon de la balle)
			percentile (float): Percentile pour estimer le fond local
		
		Returns:
			DataProcessor: Instance avec baseline corrigée
		"""
		y = self.data["intensity"].values
		L = len(y)
		
		# Estimation de la baseline par percentile glissant
		baseline = np.zeros(L)
		half_window = window // 2
		
		for i in range(L):
			start = max(0, i - half_window)
			end = min(L, i + half_window)
			baseline[i] = np.percentile(y[start:end], percentile)
		
		# Lissage de la baseline
		if window % 2 == 0:
			window += 1
		baseline = medfilt(baseline, kernel_size=window)
		
		# Soustraction
		corrected = y - baseline
		corrected = np.clip(corrected, a_min=0, a_max=None)
		
		self.data["intensity"] = corrected
		return self

	# ============================================================================
	# NOUVELLES MÉTHODES - SMOOTHING (LISSAGE)
	# ============================================================================

	def smoothing_savgol(
		self, 
		window_length: int = 11, 
		polyorder: int = 3
	) -> "DataProcessor":
		"""
		Lissage par filtre Savitzky-Golay.
		
		Préserve mieux la forme et la hauteur des pics que le filtre médian.
		Ajuste un polynôme localement pour lisser le signal.
		
		Très utilisé en spectrométrie car il préserve les moments statistiques
		(aires, hauteurs, positions des pics).
		
		Args:
			window_length (int): Taille de la fenêtre (doit être impaire)
			                    Typiquement 5-25 pour HRMS
			polyorder (int): Ordre du polynôme (2-5)
			                Ordre 2-3 recommandé pour la plupart des cas
		
		Returns:
			DataProcessor: Instance avec signal lissé
		"""
		y = self.data["intensity"].values
		
		# S'assurer que la fenêtre est impaire
		if window_length % 2 == 0:
			window_length += 1
		
		# S'assurer que polyorder < window_length
		if polyorder >= window_length:
			polyorder = window_length - 1
		
		# Application du filtre Savitzky-Golay
		smoothed = savgol_filter(y, window_length=window_length, polyorder=polyorder)
		smoothed = np.clip(smoothed, a_min=0, a_max=None)
		
		self.data["intensity"] = smoothed
		return self

	def smoothing_wavelet(
		self, 
		wavelet: str = 'sym6', 
		level: Optional[int] = None,
		threshold_method: str = 'soft'
	) -> "DataProcessor":
		"""
		Débruitage par transformée en ondelettes (Wavelet Transform).
		
		Très efficace pour séparer le signal du bruit en spectrométrie.
		Décompose le signal en différentes échelles de fréquence et applique
		un seuillage pour éliminer le bruit haute fréquence.
		
		Utilisé dans des algorithmes comme CentWave (xcms).
		
		Args:
			wavelet (str): Type d'ondelette ('sym6', 'db4', 'coif3', 'bior2.4')
			              'sym6' recommandé pour signaux spectrométriques
			level (int): Niveau de décomposition (None = auto)
			            Typiquement 3-6 pour HRMS
			threshold_method (str): 'soft' ou 'hard'
			                       'soft' donne un signal plus lisse
		
		Returns:
			DataProcessor: Instance avec signal débruité
		"""
		y = self.data["intensity"].values
		
		# Niveau de décomposition automatique si non spécifié
		if level is None:
			level = pywt.dwt_max_level(len(y), wavelet)
			level = min(level, 6)  # Limiter à 6 niveaux max
		
		# Décomposition en ondelettes
		coeffs = pywt.wavedec(y, wavelet, level=level)
		
		# Estimation du bruit (MAD - Median Absolute Deviation)
		sigma = np.median(np.abs(coeffs[-1])) / 0.6745
		
		# Seuil universel de VisuShrink
		threshold = sigma * np.sqrt(2 * np.log(len(y)))
		
		# Application du seuillage sur tous les niveaux sauf les approximations
		coeffs_thresh = [coeffs[0]]  # Garder les approximations (signal basse fréq)
		for i in range(1, len(coeffs)):
			coeffs_thresh.append(
				pywt.threshold(coeffs[i], threshold, mode=threshold_method)
			)
		
		# Reconstruction du signal
		denoised = pywt.waverec(coeffs_thresh, wavelet)
		
		# Ajustement de la taille (peut différer légèrement)
		if len(denoised) > len(y):
			denoised = denoised[:len(y)]
		elif len(denoised) < len(y):
			denoised = np.pad(denoised, (0, len(y) - len(denoised)), mode='edge')
		
		denoised = np.clip(denoised, a_min=0, a_max=None)
		self.data["intensity"] = denoised
		return self

	def smoothing_whittaker(
		self, 
		lam: float = 1e5
	) -> "DataProcessor":
		"""
		Lissage de Whittaker (Penalized Least Squares).
		
		Méthode de lissage très flexible qui équilibre fidélité aux données
		et lissage via un paramètre de pénalisation.
		
		Similaire à ALS mais sans asymétrie (traite pics et vallées de manière égale).
		
		Args:
			lam (float): Paramètre de lissage (10^2 à 10^8)
			            Plus grand = plus lisse
			            Typiquement 10^4 - 10^6 pour HRMS
		
		Returns:
			DataProcessor: Instance avec signal lissé
		"""
		y = self.data["intensity"].values
		L = len(y)
		
		# Matrice de différences de second ordre
		D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
		
		# Résolution du système linéaire
		W = sparse.eye(L)
		Z = W + lam * D.dot(D.transpose())
		z = spsolve(Z, y)
		
		z = np.clip(z, a_min=0, a_max=None)
		self.data["intensity"] = z
		return self
	
	# 1 ROI
	def filter_roi(self, min_points=3) -> "DataProcessor":
		df = self.data.copy()
		mask = df.groupby('rt')['intensity'].transform(lambda x: (x>0).sum() >= min_points)
		self.data = df[mask].reset_index(drop=True)
		return self
	# 2 Gaussien
	def smoothing_gaussian(self, sigma=2) -> "DataProcessor":
		df = self.data.copy()
		df['intensity'] = gaussian_filter1d(df['intensity'].values, sigma=sigma)
		self.data = df
		return self
	# 3 Filter Median
	def smoothing_median(self, kernel_size=5) -> "DataProcessor":
		df = self.data.copy()
		df['intensity'] = medfilt(df['intensity'].values, kernel_size=kernel_size)
		self.data = df
		return self


	# ============================================================================
	# MÉTHODES ORIGINALES (NOISE REDUCTION)
	# ============================================================================

	def background_noise_correction(
		self,
		factor: float = 3.0,
		segment_size: int = 100,
		variation_threshold: float = 0.05
	) -> "DataProcessor":
		"""
		Applique une correction du bruit de fond sur la colonne 'intensity' via une analyse par segments.

		La méthode découpe les données en segments de taille fixe, identifie ceux à faible variation
		(considérés comme du bruit) puis estime un seuil d'intensité à partir de leur écart-type moyen.
		Toute valeur inférieure à ce seuil est remplacée par zéro.

		Args:
			factor (float): Multiplicateur du seuil de bruit (basé sur l'écart-type).
			segment_size (int): Taille (en nombre de points) de chaque segment à analyser.
			variation_threshold (float): Seuil maximal de variation relative pour qu'un segment soit considéré comme bruit.

		Returns:
			DataProcessor: L'instance actuelle après correction du bruit.
		"""
		# === Étape 1 : Préparation des données ===
		intensity = self.data["intensity"].to_numpy(dtype=float, copy=True)
		total_points = len(intensity)
		n_segments = total_points // segment_size

		# Vérification : il faut au moins un segment complet
		if n_segments == 0:
			print("Aucun segment complet détecté. Veuillez diminuer segment_size.")
			return self

		# === Étape 2 : Découpage en segments ===
		# On ignore les points restants qui ne forment pas un segment complet
		segments = intensity[:n_segments * segment_size].reshape((n_segments, segment_size))

		# === Étape 3 : Calcul des statistiques par segment ===
		seg_max = segments.max(axis=1)
		seg_min = segments.min(axis=1)
		seg_std = segments.std(axis=1)

		# Variation relative = (max - min) / max (ajusté pour éviter la division par zéro)
		variations = (seg_max - seg_min) / (seg_max + 1e-9)

		# === Étape 4 : Identification des segments à faible variation (bruit) ===
		noise_std = seg_std[variations <= variation_threshold]

		# Moyenne des écarts-types des segments bruités (σµ)
		sigma_mu = noise_std.mean() if noise_std.size > 0 else 0.0

		# === Étape 5 : Suppression du bruit — toute intensité < σµ * facteur est mise à zéro ===
		self.data["intensity"] = self.data["intensity"].apply(
			lambda x: 0.0 if x < (sigma_mu * factor) else x
		)

		# Retour de l'instance pour chaînage
		return self

	# ============================================================================
	# NOUVELLES MÉTHODES - NOISE REDUCTION AVANCÉE
	# ============================================================================

	def noise_reduction_mad(
		self, 
		factor: float = 3.0,
		window: int = 100
	) -> "DataProcessor":
		"""
		Réduction du bruit par MAD (Median Absolute Deviation).
		
		Méthode robuste aux outliers qui estime le niveau de bruit
		via la déviation absolue médiane, puis applique un seuillage.
		
		Plus robuste que l'écart-type standard pour données bruitées.
		
		Args:
			factor (float): Multiplicateur du seuil (typiquement 2-5)
			window (int): Taille de fenêtre pour estimation locale du bruit
		
		Returns:
			DataProcessor: Instance avec bruit réduit
		"""
		y = self.data["intensity"].values
		L = len(y)
		
		# Calcul du MAD local
		half_window = window // 2
		threshold_array = np.zeros(L)
		
		for i in range(L):
			start = max(0, i - half_window)
			end = min(L, i + half_window)
			segment = y[start:end]
			
			# MAD = médiane des déviations absolues par rapport à la médiane
			median = np.median(segment)
			mad = np.median(np.abs(segment - median))
			
			# Conversion MAD -> écart-type équivalent
			sigma_equiv = 1.4826 * mad
			threshold_array[i] = factor * sigma_equiv
		
		# Application du seuil
		denoised = np.where(y > threshold_array, y, 0)
		
		self.data["intensity"] = denoised
		return self

	def noise_reduction_percentile(
		self, 
		percentile: float = 95.0,
		window: int = 100
	) -> "DataProcessor":
		"""
		Réduction du bruit par seuillage basé sur percentiles locaux.
		
		Conserve uniquement les valeurs au-dessus d'un certain percentile
		calculé dans une fenêtre glissante.
		
		Args:
			percentile (float): Percentile de coupure (90-99)
			                   95 = garde les 5% de valeurs les plus élevées
			window (int): Taille de la fenêtre pour calcul local
		
		Returns:
			DataProcessor: Instance avec bruit réduit
		"""
		y = self.data["intensity"].values
		L = len(y)
		
		half_window = window // 2
		threshold_array = np.zeros(L)
		
		for i in range(L):
			start = max(0, i - half_window)
			end = min(L, i + half_window)
			threshold_array[i] = np.percentile(y[start:end], percentile)
		
		# Conserver uniquement les valeurs au-dessus du seuil
		denoised = np.where(y > threshold_array, y, 0)
		
		self.data["intensity"] = denoised
		return self

	# ============================================================================
	# MÉTHODES COMBINÉES - PIPELINES
	# ============================================================================

	def preprocess_pipeline_basic(self) -> "DataProcessor":
		"""
		Pipeline de prétraitement BASIQUE (méthodes originales).
		
		1. Correction baseline (médian)
		2. Correction bruit de fond (segments)
		
		Returns:
			DataProcessor: Instance prétraitée
		"""
		self.baseline_correction(window=50)
		self.background_noise_correction(factor=3.0)
		return self

	def preprocess_pipeline_advanced(self) -> "DataProcessor":
		"""
		Pipeline de prétraitement AVANCÉ (méthodes optimisées).
		
		1. Correction baseline ALS (meilleure que médian)
		2. Lissage Savitzky-Golay (préserve les pics)
		3. Débruitage wavelet (élimine bruit haute fréquence)
		4. Seuillage MAD (robuste)
		
		Returns:
			DataProcessor: Instance prétraitée
		"""
		self.baseline_als(lam=1e6, p=0.01)
		self.smoothing_savgol(window_length=11, polyorder=3)
		self.smoothing_wavelet(wavelet='sym6', level=4)
		self.noise_reduction_mad(factor=3.0)
		return self

	def preprocess_pipeline_gentle(self) -> "DataProcessor":
		"""
		Pipeline de prétraitement DOUX (minimal, préserve au max le signal).
		
		1. Correction baseline ALS
		2. Lissage léger Savitzky-Golay
		
		Returns:
			DataProcessor: Instance prétraitée
		"""
		self.baseline_als(lam=1e5, p=0.05)
		self.smoothing_savgol(window_length=7, polyorder=2)
		return self

	def preprocess_pipeline_aggressive(self) -> "DataProcessor":
		"""
		Pipeline de prétraitement AGRESSIF (max réduction de bruit).
		
		1. Baseline top-hat (morphologique)
		2. Wavelet fort
		3. Whittaker smoothing
		4. Double seuillage (MAD + percentile)
		
		Returns:
			DataProcessor: Instance prétraitée
		"""
		self.baseline_tophat(structure_size=100)
		self.smoothing_wavelet(wavelet='sym6', level=5, threshold_method='hard')
		self.smoothing_whittaker(lam=1e6)
		self.noise_reduction_mad(factor=4.0)
		self.noise_reduction_percentile(percentile=97.0)
		return self

	# ============================================================================
	# MÉTHODES ORIGINALES (FILTRAGE)
	# ============================================================================

	def filter_SignalNoise(
		self,
		sort_column: Union[list, str] = ["rt"],
		window_size: int = 5,
		margin: int = 2,
		factor: int = 3
	) -> "DataProcessor":
		"""
		Applique un filtrage basé sur le rapport Signal/Bruit (S/N) pour sélectionner les pics significatifs.

		Cette méthode utilise une approche locale pour estimer le bruit dans une fenêtre centrée sur chaque
		maximum local. Seuls les points ayant un rapport S/N supérieur ou égal à un seuil donné sont conservés.

		Args:
			sort_column (Union[list, str]): Colonne(s) utilisée(s) pour trier les données (ex : 'mz' ou ['rt']).
			window_size (int): Taille de la fenêtre de bruit autour du pic (hors marge).
			margin (int): Nombre de points à exclure autour du pic pour éviter toute influence.
			factor (int): Seuil minimal du rapport signal/bruit (ex : 3 = S/N ≥ 3).

		Returns:
			DataProcessor: L'instance modifiée, avec uniquement les pics respectant le critère S/N.
		"""

		# Application du filtre S/N via la fonction externe `filter_SN`
		self.data = filter_SN(
			df=self.data,
			sort_column=sort_column,
			window_size=window_size,
			margin=margin,
			factor=factor
		)

		# Conversion explicite des colonnes en float (sécurité typage)
		self.data = self.data.astype({col: float for col in ["rt", "mz", "dt", "intensity"]})

		# Retour de l'instance pour permettre un chaînage fluide
		return self

	def filter_by_column(
		self,
		filters: Optional[dict[str, tuple[float, float]]] = None
	) -> "DataProcessor":
		"""
		Applique des filtres sur les colonnes numériques du DataFrame selon des bornes personnalisées.

		Chaque filtre consiste en une plage de valeurs (min, max) pour une colonne donnée :
			- Si min est None → pas de borne inférieure.
			- Si max est None → pas de borne supérieure.

		Args:
			filters (dict[str, tuple[float | None, float | None]]): 
				Dictionnaire des filtres à appliquer.
				Exemple :
					{
						"mz": (100, 200),        # mz compris entre 100 et 200
						"rt": (10.3, None),      # rt > 10.3 uniquement
						"intensity": (None, 50)  # intensity < 50 uniquement
					}

		Returns:
			DataProcessor: L'instance actuelle après filtrage.
		"""
		# === Étape 1 : Aucun filtre fourni → retour des données inchangées
		if filters is None:
			return self

		# === Étape 2 : Application des filtres colonne par colonne
		for column, (min_value, max_value) in filters.items():
			# Vérification de l'existence de la colonne
			if column not in self.data.columns:
				raise ValueError(f"La colonne '{column}' est absente des données.")

			# Application du filtre inférieur (> min)
			if min_value is not None:
				self.data = self.data[self.data[column] > min_value]

			# Application du filtre supérieur (< max)
			if max_value is not None:
				self.data = self.data[self.data[column] < max_value]

		# === Étape 3 : Réinitialisation de l'index pour garder un DataFrame propre
		self.data = self.data.reset_index(drop=True)

		# === Étape 4 : Retour de l'instance pour chaînage
		return self

	# ============================================================================
	# MÉTHODES D'ÉVALUATION ET COMPARAISON
	# ============================================================================

	def compute_metrics(self) -> dict:
		"""
		Calcule des métriques de qualité sur les données actuelles.
		
		Permet de comparer objectivement différents prétraitements.
		
		Returns:
			dict: Dictionnaire contenant:
				- 'n_points': Nombre de points non-nuls
				- 'mean_intensity': Intensité moyenne (points > 0)
				- 'std_intensity': Écart-type de l'intensité
				- 'noise_estimate': Estimation du niveau de bruit (MAD)
				- 'snr_estimate': Estimation du rapport signal/bruit
				- 'dynamic_range': Plage dynamique (max/min non-nul)
		"""
		intensity = self.data["intensity"].values
		non_zero = intensity[intensity > 0]
		
		if len(non_zero) == 0:
			return {
				'n_points': 0,
				'mean_intensity': 0,
				'std_intensity': 0,
				'noise_estimate': 0,
				'snr_estimate': 0,
				'dynamic_range': 0
			}
		
		# Estimation du bruit par MAD
		median = np.median(non_zero)
		mad = np.median(np.abs(non_zero - median))
		noise = 1.4826 * mad
		
		# SNR estimé
		signal = np.mean(non_zero)
		snr = signal / noise if noise > 0 else 0
		
		# Plage dynamique
		dynamic_range = np.max(non_zero) / np.min(non_zero) if np.min(non_zero) > 0 else 0
		
		return {
			'n_points': len(non_zero),
			'mean_intensity': np.mean(non_zero),
			'std_intensity': np.std(non_zero),
			'noise_estimate': noise,
			'snr_estimate': snr,
			'dynamic_range': dynamic_range
		}

	def get_comparison_data(self) -> pd.DataFrame:
		"""
		Retourne un DataFrame avec les données originales et traitées côte à côte.
		
		Utile pour visualisation comparative.
		
		Returns:
			pd.DataFrame: DataFrame avec colonnes '_original' et '_processed'
		"""
		comparison = self.original_data.copy()
		comparison['intensity_original'] = self.original_data['intensity']
		comparison['intensity_processed'] = self.data['intensity']
		
		return comparison
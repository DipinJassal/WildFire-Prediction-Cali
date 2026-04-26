import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Post-hoc isotonic calibration wrapper (replaces cv='prefit' removed in sklearn 1.4)."""
    def __init__(self, model):
        self.estimator   = model   # exposed for feature_importances_ access
        self._calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X_cal, y_cal):
        raw = self.estimator.predict_proba(X_cal)[:, 1]
        self._calibrator.fit(raw, y_cal)
        return self

    def predict_proba(self, X):
        raw = self.estimator.predict_proba(X)[:, 1]
        cal = self._calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EnsembleClassifier:
    """Soft-voting ensemble: weighted average of predict_proba from multiple models."""
    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights or [1 / len(models)] * len(models)

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# Approximate mean elevations (metres) derived from SRTM/NED for all 58 CA counties
COUNTY_ELEVATION = {
    "Alameda County":        100,
    "Alpine County":        2200,
    "Amador County":         800,
    "Butte County":          400,
    "Calaveras County":      900,
    "Colusa County":          50,
    "Contra Costa County":   100,
    "Del Norte County":      300,
    "El Dorado County":     1200,
    "Fresno County":         300,
    "Glenn County":          100,
    "Humboldt County":       400,
    "Imperial County":       -20,
    "Inyo County":          1500,
    "Kern County":           600,
    "Kings County":           75,
    "Lake County":           600,
    "Lassen County":        1500,
    "Los Angeles County":    300,
    "Madera County":         600,
    "Marin County":          200,
    "Mariposa County":      1200,
    "Mendocino County":      500,
    "Merced County":          75,
    "Modoc County":         1400,
    "Mono County":          2000,
    "Monterey County":       400,
    "Napa County":           300,
    "Nevada County":        1400,
    "Orange County":         100,
    "Placer County":        1000,
    "Plumas County":        1500,
    "Riverside County":      500,
    "Sacramento County":      20,
    "San Benito County":     400,
    "San Bernardino County": 900,
    "San Diego County":      300,
    "San Francisco County":   50,
    "San Joaquin County":     20,
    "San Luis Obispo County":300,
    "San Mateo County":      200,
    "Santa Barbara County":  400,
    "Santa Clara County":    100,
    "Santa Cruz County":     300,
    "Shasta County":         800,
    "Sierra County":        1800,
    "Siskiyou County":      1200,
    "Solano County":          75,
    "Sonoma County":         300,
    "Stanislaus County":      75,
    "Sutter County":          20,
    "Tehama County":         300,
    "Trinity County":       1200,
    "Tulare County":         800,
    "Tuolumne County":      1600,
    "Ventura County":        400,
    "Yolo County":            30,
    "Yuba County":           300,
}

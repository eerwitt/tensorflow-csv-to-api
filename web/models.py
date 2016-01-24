class Iris(object):
    def __init__(self, species=None, features=None):
        self.species = species

        if features is not None:
            self._sepal_length = features[0]
            self._sepal_width = features[1]
            self._petal_length = features[2]
            self._petal_width = features[3]
        else:
            self._sepal_length = None
            self._sepal_width = None
            self._petal_length = None
            self._petal_width = None

    def _asdict(self):
        return {
            'species': self.species,
            'sepal_length': self.sepal_length,
            'sepal_width': self.sepal_width,
            'petal_length': self.petal_length,
            'petal_width': self.petal_width
        }

    @property
    def sepal_length(self):
        return self._sepal_length

    @property
    def sepal_width(self):
        return self._sepal_width

    @property
    def petal_length(self):
        return self._petal_length

    @property
    def petal_width(self):
        return self._petal_width


class Prediction(object):
    def __init__(self, iris, y):
        self._iris = iris
        self._y = y

    def _asdict(self):
        return {
            'iris': self.iris._asdict(),
            'y': self.y
        }

    @property
    def iris(self):
        return self._iris

    @property
    def y(self):
        return self._y


class PredictionRequest(object):
    def __init__(self, iris_features):
        self._uuid = None
        self._prediction = None
        self._status = "pending"
        self._iris_features = iris_features

    def _asdict(self):
        prediction = None
        if self.prediction:
            prediction = self.prediction._asdict()

        return {
            'uuid': self.uuid,
            'status': self.status,
            'iris_features': self.iris_features._asdict(),
            'prediction': prediction
        }

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = value

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value
        self._status = "fulfilled"

    @property
    def status(self):
        return self._status

    @property
    def iris_features(self):
        return self._iris_features

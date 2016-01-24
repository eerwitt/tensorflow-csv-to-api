from iris import network
from iris import dataset

from web.models import Iris
from web.models import Prediction
from web.models import PredictionRequest

import simplejson as json

import falcon
import uuid


class PredictionRequestStorageEngine(object):
    """
    To avoid extra dependencies, this is a fake storage engine based on the
    Falcon example docs.

    Notes
    -----
    Never use this in production, it is in memory only!

    See Also
    --------
    http://falcon.readthedocs.org/en/latest/user/quickstart.html
    """
    def __init__(self):
        self._store = {}

    def get_prediction_request(self, prediction_request_uuid):
        """
        Get a prediction request based on its UUID.

        Parameters
        ----------
        prediction_request_uuid : str
            UUID for the requested PredictionRequest.

        Returns
        -------
        prediction_request : PredictionRequest
            The PredictionRequest with the requested UUID.
        """
        return self._store[prediction_request_uuid]

    def add_prediction_request(self, prediction_request):
        """
        "Save" a prediction request into our in memory fake storage.

        Parameters
        ----------
        prediction_request : PredictionRequest
            An Iris with features but no species to predict its species from.

        Returns
        -------
        prediction_request : PredictionRequest
            The prediction request which was just "saved".
        """
        prediction_request_uuid = str(uuid.uuid4())
        prediction_request.uuid = prediction_request_uuid
        self._store[prediction_request_uuid] = prediction_request

        return prediction_request


def allow_swagger_editor(req, res, resource):
    """
    Allow cross origin requests from the Swagger Online editor at:
        http://editor.swagger.io/

    Parameters
    ----------
    req : FalconRequest
        Request being made through this Falcon middleware.
    res : FalconResponse
        Response to send back to clients from this Falcon middleware.
    resource : FalconResource
        Resource being requested.
    """
    res.set_header(
        "Access-Control-Allow-Origin", "http://editor.swagger.io")
    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    res.set_header("Access-Control-Allow-Headers", "Content-Type")


class PredictionRequestsResource(object):
    """
    List resource for PredictionRequests.
    """
    def __init__(self, db):
        self._db = db

    def on_post(self, req, res):
        """
        Create a PredictionRequest based on the JSON body.

        See ./swagger.json for details.
        """
        body = json.loads(req.stream.read().decode('utf8'))
        iris_test = Iris(features=[
            body["sepal_length"],
            body["sepal_width"],
            body["petal_length"],
            body["petal_width"]])
        prediction_request = PredictionRequest(
            iris_features=iris_test)

        self._db.add_prediction_request(prediction_request)

        res.status = falcon.HTTP_CREATED
        res.body = json.dumps(
            prediction_request._asdict(),
            use_decimal=True)


class PredictionRequestResource(object):
    def __init__(self, db, onehot_species, net, sess):
        self._db = db
        self._onehot_species = onehot_species

        # Note, this session won't cleanup after itself without restarting the
        # webserver.
        self.net = net
        self.sess = sess

    def on_get(self, req, res, prediction_uuid):
        """
        Get the results from a prediction based on a prediction request's
        features.

        See ./swagger.json for details.
        """
        prediction_request = self._db.get_prediction_request(prediction_uuid)

        iris_test = prediction_request.iris_features

        y = network.predict_with_session(
            self.net,
            [[
                iris_test.sepal_length,
                iris_test.sepal_width,
                iris_test.petal_length,
                iris_test.petal_width
            ]],
            self.sess)

        predicted_onehot = network.onehot_from_argmax(y[1])
        predicted_species = self._onehot_species[predicted_onehot]

        iris_found = Iris(species=predicted_species)

        prediction_tensor = list(map(lambda p: round(float(p), 4), y[0][0]))
        prediction = Prediction(
            iris=iris_found,
            y=prediction_tensor)

        prediction_request.prediction = prediction

        res.status = falcon.HTTP_OK
        res.body = json.dumps(
            prediction_request._asdict(),
            use_decimal=True)


# Global session for predicting with, if we reuse sessions then the checkpoint
# restore will create duplicate variables and fail.
sess, net = network.predict_init("./checkpoints")
db = PredictionRequestStorageEngine()

onehot_species = {}
for species, onehot in dataset.read_species_onehot_csv("./raw"):
    onehot_species[onehot] = species

api = falcon.API(after=[allow_swagger_editor])
api.add_route(
    '/predictionrequest/',
    PredictionRequestsResource(db))

api.add_route(
    '/predictionrequest/{prediction_uuid}',
    PredictionRequestResource(db, onehot_species, net, sess))

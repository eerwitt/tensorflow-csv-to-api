from network import iris

from web.models import Iris
from web.models import Prediction
from web.models import PredictionRequest

import simplejson as json

import falcon
import uuid


class PredictionRequestStorageEngine(object):
    def __init__(self):
        self._store = {}

    def get_prediction_request(self, prediction_request_uuid):
        return self._store[prediction_request_uuid]

    def add_prediction_request(self, prediction_request):
        prediction_request_uuid = str(uuid.uuid4())
        prediction_request.uuid = prediction_request_uuid
        self._store[prediction_request_uuid] = prediction_request

        return prediction_request


def allow_swagger_editor(req, res, resource):
    res.set_header(
        "Access-Control-Allow-Origin", "http://editor.swagger.io")
    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    res.set_header("Access-Control-Allow-Headers", "Content-Type")


class PredictionRequestsResource(object):
    def __init__(self, db):
        self._db = db

    def on_post(self, req, res):
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
    def __init__(self, db, net, sess):
        self._db = db

        # Note, this session won't cleanup after itself without restarting the
        # webserver.
        self.net = net
        self.sess = sess

    def on_get(self, req, res, prediction_uuid):
        prediction_request = self._db.get_prediction_request(prediction_uuid)

        iris_found = Iris(species="Iris Setosa")
        iris_test = prediction_request.iris_features

        y = iris.predict_with_session(
            self.net,
            [[
                iris_test.sepal_length,
                iris_test.sepal_width,
                iris_test.petal_length,
                iris_test.petal_width
            ]],
            self.sess)

        prediction = Prediction(
            iris=iris_found,
            y=list(map(lambda p: round(float(p), 4), y[0][0])))

        prediction_request.prediction = prediction

        res.status = falcon.HTTP_OK
        res.body = json.dumps(
            prediction_request._asdict(),
            use_decimal=True)


# Global session for predicting with, if we reuse sessions then the checkpoint
# restore will create duplicate variables and fail.
sess, net = iris.predict_init("./checkpoints")
db = PredictionRequestStorageEngine()

api = falcon.API(after=[allow_swagger_editor])
api.add_route(
    '/predictionrequest/',
    PredictionRequestsResource(db))

api.add_route(
    '/predictionrequest/{prediction_uuid}',
    PredictionRequestResource(db, net, sess))

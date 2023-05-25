from flask import Flask, request
from flask_api import status
from flask_restful import Api, Resource

from custom_flask_api import CustomFlaskApi
from scripts import model_initialize

flask_app = Flask(__name__)

model, kmeans, X_encoded, knn, data, files = model_initialize()

environment = flask_app.config["ENV"]
environment = f"config.Config"
flask_app.config.from_object(environment)

api = CustomFlaskApi(flask_app)


class XGenAPIView(Resource):
    def post(self):
        request_data = request.files
        image = request_data.get("image", None)
        params_valid = False

        if image:
            params_valid = self.validate_params(image)

        if not params_valid:
            return {
                "message": "Invalid params. Please provide image in params"
            }, status.HTTP_400_BAD_REQUEST

        return {"data": self.predict(image.filename)}, status.HTTP_200_OK

    def predict(self, image):
        """
        Predict the 10 similar images according to the image
        Arguments:
        image - image filename
        """
        try:
            num = files.index(image)
        except ValueError:
            return "Not found"

        result = knn.kneighbors(
            data[num].reshape(1, -1), return_distance=True, n_neighbors=10
        )
        result_array = list(result[1][0])[1:]
        result_names = [
            files[i] for i in result_array]
        return {"input": files[num], "output": result_names}

    def validate_params(self, file):
        params_valid = False
        if file.content_type in ["image/jpeg", "image/png"]:
            params_valid = True
        return params_valid


api.add_resource(XGenAPIView, "/predict")


if __name__ == "__main__":
    flask_app.run()

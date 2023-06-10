from locust import HttpUser, task
import json

class HelloWorldUser(HttpUser):
    @task
    def phase1_prob1(self):
        with open('./phase1_prob1_rows.json', 'r') as f:
            self.client.post("phase-1/prob-1/predict", json={
                "id": "first",
                "columns": [
                    "feature1", "feature2",  "feature3", "feature4", "feature5", "feature6", "feature7", "feature8",
                    "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"
                ],
                "rows": json.load(f)
            })

    @task
    def phase1_prob2(self):
        with open('./phase1_prob2_rows.json', 'r') as f:
            self.client.post("phase-1/prob-2/predict", json={
                "id": "first",
                "columns": [
                    "feature1", "feature2",  "feature3", "feature4", "feature5", "feature6", "feature7", "feature8",
                    "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16",
                    "feature17", "feature18", "feature19", "feature20"
                ],
                "rows": json.load(f)
            })

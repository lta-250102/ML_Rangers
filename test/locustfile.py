from locust import HttpUser, task

class HelloWorldUser(HttpUser):
    # @task
    # def hello_world(self):
    #     self.client.get("")

    @task
    def phase1_prob1(self):
        self.client.post("1/1/predict", json={
            "id": "first",
            "columns": [
                "feature1", "feature2",  "feature3", "feature4", "feature5", "feature6", "feature7", "feature8",
                "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16",
            ],
            "rows": [
                ["Site engineer", "grocery_pos", 8.6, 48230, 40.21343879339888, -85.2037563034635, 47583, 42.508293, -83.168004, 65.59606217585437, 3, 5, 1, 8.017864754614141, 1.0288222577545105, 58.91113204988728],
                ["Site engineer", "gas_transport", 316.84, 48230, 44.37939089316718, -82.85972140937571, 47583, 42.661838, -81.96651, 64.72879520544782, 6, 5, 1, 11.768567521927777, 1.1062168564714336, 64.43101705524653],
            ]
        })

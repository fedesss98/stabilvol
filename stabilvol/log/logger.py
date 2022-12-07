import neptune.new as neptune


class Logger(dict):

    def upload(self):
        run = neptune.init(
            project="federico.amato/stabilizing-volatility",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZGRmMzMwNy03MTU1LTQ3NTItYWZkZC0yYTkxYzFiYWRlYTUifQ==",
        )  # your credentials

        params = {"learning_rate": 0.001, "optimizer": "Adam"}
        run["parameters"] = params

        for epoch in range(10):
            run["train/loss"].log(0.9 ** epoch)

        run["eval/f1_score"] = 0.66

        run.stop()

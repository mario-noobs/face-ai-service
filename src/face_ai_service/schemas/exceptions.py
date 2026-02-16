class FaceAIException(Exception):
    def __init__(self, message: dict, detail: str = None):
        self.code = message["code"]
        self.message = message["message"]
        self.detail = detail
        super().__init__(self.message)

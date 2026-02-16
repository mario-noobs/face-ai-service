class Messages:
    SUCCESS = {"code": "0000", "message": "Operation successful."}
    MISSING_FIELDS = {"code": "4000", "message": "Required fields are missing."}
    MISSING_ALG = {"code": "4002", "message": "Algorithm is missing."}
    IMAGE_BASE64_ERROR = {"code": "4001", "message": "Failed to decode the image."}
    GENERIC_ERROR = {"code": "1111", "message": "An error occurred while processing the request."}
    IMAGE_ENCODING_ERROR = {"code": "5001", "message": "Error creating face encoding."}
    NO_FACE = {"code": "5002", "message": "No face detected in the image."}
    MODEL_NOT_LOADED = {"code": "5003", "message": "Requested model is not loaded."}
    INVALID_CANDIDATES = {"code": "4003", "message": "Invalid or empty candidates list."}

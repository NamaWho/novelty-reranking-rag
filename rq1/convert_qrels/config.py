class Config:
    def __init__(self,
                 encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 threshold: float = 0.85,
                    ):
        self.encoder_name = encoder_name
        self.threshold = threshold
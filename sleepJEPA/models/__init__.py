from .jepa_encoder import JEPAModel, EEG1DTransformerEncoder, Predictor

# Backward compatibility: export `JEPA` name pointing to the model class used elsewhere.
JEPA = JEPAModel

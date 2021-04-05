import pandas as pd
import joblib
import sklearn.metrics as met


y_col = 'arr_delay'
submission_columns = [
    'fl_date',
    'mkt_carrier',
    'mkt_carrier_fl_num',
    'origin',
    'dest'
]


def clean_train(x):
    """
    Cleans the training data (e.g. removing nulls and outliers).
    """
    x_no_na = x.dropna(
        subset=[
            'tail_num', 'dep_time', 'dep_delay', 'taxi_out',
            'wheels_off', 'wheels_on', 'taxi_in', 'arr_time',
            'arr_delay', 'actual_elapsed_time', 'air_time'
        ]
    )
    
    return clean_test(x_no_na)


def clean_test(x):
    """
    Cleans the testing data. This can't remove
    rows, but can do other cleaning like
    standardizing representations and imputing
    missing values. 
    """
    return x


def save(model, path):
    """
    Saves the specified trained model to the specified
    file path.
    """
    joblib.dump(model, path + '.joblib')


def load(path):
    """
    Loads the trained model from the specified file path.
    """
    return joblib.load(path + '.joblib')

    
def extract_x_y(xy, x_transformer, y_transformer=None):
    """
    Extracts the (transformed) x and y variables from a dataframe containing both.
    """
    x = x_transformer(xy.drop(y_col, axis=1))
    y = xy[[y_col]]
    if y_transformer:
        y = y_transformer(y)
    return x, y


class Model:
    def __init__(
        self, model, x_transformer, y_untransformer=None
    ):
        """
        Creates a Model using the specified sklearn model and transformers.
        
        The x transformer is a function that transforms the independent
        variables dataframe into the form expected by the model.
        This includes throwing away unused columns, creating
        split or combined columns, scaling and normalizing data,
        and converting categorical columns to numeric columns.
        
        The y untransformer is a function that transforms the model's
        predictions into something comparable against the target
        values; for example, if the model is trained against
        target values with a Box-Cox transform applied, this
        function must apply the inverse transform. If None,
        the model's predictions are taken directly.
        """
        self.model = model
        self.x_transformer = x_transformer
        self.y_untransformer = y_untransformer or (lambda x: x)

    def validate(self, valid_x, valid_y):
        """
        Validates the specified model by predicting y values
        from the specified x dataframe. Returns the
        r-squared score for the predictions.
        """
        predictions = self.predict(valid_x)
        return met.r2_score(valid_y, predictions)

    def predict(self, x):
        """
        Predicts y values for the specified data.
        """
        transformed_x = self.x_transformer(x).values
        predictions = self.model.predict(transformed_x)
        return self.y_untransformer(predictions)

    def submit(self, x_path, submission_path, y_column_name):
        """
        Creates the submission file by predicting y
        values for the x values at x_path.
        """
        x = pd.read_csv(x_path)
        predictions = self.predict(x)
        x = x[submission_columns]
        x[y_column_name] = predictions
        x.to_csv(submission_path, index=False)


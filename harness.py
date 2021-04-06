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


def normalize(y):
    normalizer = load('y_normalizer')
    y_tr = normalizer.transform(y)
    y_tr = pd.DataFrame(y_tr, index=y.index, columns=y.columns)
    return y_tr


def unnormalize(y_tr):
    normalizer = load('y_normalizer')
    y = normalizer.inverse_transform(y_tr)
    y = pd.DataFrame(y, index=y_tr.index, columns=y_tr.columns)
    return y


class DataTransformer:
    def __init__(self, x_transformer, y_transformer=None, y_untransformer=None):
        """
        Creates a DataTransformer with the specified transformers.
        
        The x transformer is a function that transforms the independent
        variables dataframe into the form expected by the model.
        This includes throwing away unused columns, creating
        split or combined columns, scaling and normalizing data,
        and converting categorical columns to numeric columns.
        
        The y transformer is a function that transforms the dependent
        variables (i.e. training labels) into the form needed to train
        the model. For example, this might involve normalizing the
        values or applying a Box-Cox transform.
        
        The y untransformer is a function that transforms the model's
        predictions into something comparable against the target
        values. This should be the inverse of the y transformer.
        """
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer or (lambda y: y)
        self.y_untransformer = y_untransformer or (lambda y: y)
    
    @staticmethod
    def extract_x_y(xy):
        """
        Separates a dataframe containing both the independent and
        dependent variables into two dataframes, one for the
        independent variables and one for the dependent variables.
        """
        x = xy.drop(y_col, axis=1)
        y = xy[[y_col]]
        return x, y
    
    def extract_transform(self, xy):
        """
        Separates the dataframe into the independent and dependent
        variables and applies the x and y transformers.
        """
        x, y = self.extract_x_y(xy)
        return self.x_transformer(x), self.y_transformer(y)
    
    
class TrainedModel:
    def __init__(self, model, transformers):
        """
        Creates a TrainedModel with the specified trained sklearn model
        and transformers.
        """
        self.model = model
        self.transformers = transformers
    
    def validate(self, valid_xy):
        """
        Validates the specified model by predicting y values
        from the specified (untransformed) x dataframe. Returns the
        r-squared score for the predictions.
        """
        x_tr, y_tr = self.transformers.extract_transform(valid_xy)
        predictions = self.model.predict(x_tr)
        return met.r2_score(y_tr, predictions)
    
    def submit(self, x_path, submission_path, y_column_name):
        """
        Creates the submission file by predicting (untransformed) y
        values for the (untransformed) x values at x_path.
        """
        x = pd.read_csv(x_path)
        x_tr = self.transformers.x_transformer(x)
        y_tr = self.model.predict(x_tr)
        y_tr = pd.DataFrame(y_tr, index=x.index, columns=[y_column_name])
        y = self.transformers.y_untransformer(y_tr)
        x = x[submission_columns]
        x[y_column_name] = y
        x.to_csv(submission_path, index=False)
        

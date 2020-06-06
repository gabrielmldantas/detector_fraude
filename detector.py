from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, LongType, StructType, StructField, StringType, DecimalType
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel

TRANSACTION_TYPE_MAPPING = {
    'PAYMENT': 0,
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
}

CLIENT_TYPE_MAPPING = {
    'C': '1',
    'M': '2'
}

class Detector:
    def __init__(self):
        self.spark = None
        self.model_path = Path('data/model.parquet')
        self.load_events_schema()
        self.connect()

    def connect(self):
        if not self.spark:
            self.spark = SparkSession.builder.master('spark://localhost:7077').appName('Detector de Fraudes').config('spark.executor.memory', '8g').getOrCreate()

    def stop(self):
        if self.spark:
            self.spark.stop()
        self.spark = None

    def load_events_schema(self):
        self.events_schema = StructType(fields=[
            StructField('type', IntegerType(), False),
            StructField('amount', DecimalType(), False),
            StructField('oldbalanceOrg', DecimalType(), False),
            StructField('newbalanceOrig', DecimalType(), False),
            StructField('oldbalanceDest', DecimalType(), False),
            StructField('newbalanceDest', DecimalType(), False)
        ])

    def load_data(self):
        prepared_data_path = Path('data/prepared_data.parquet')
        if prepared_data_path.exists():
            return self.spark.read.parquet(str(prepared_data_path))

        convert_type = udf(lambda type_: TRANSACTION_TYPE_MAPPING[type_], IntegerType())
        convert_name = udf(lambda name: int(CLIENT_TYPE_MAPPING[name[0]]), IntegerType())

        data = self.spark.read.load('data/PS_20174392719_1491204439457_log.csv', format='csv', sep=',', inferSchema='true', header='true')
        data = data.filter(data.isFlaggedFraud == 0)\
            .drop(data.isFlaggedFraud)\
            .drop(data.nameOrig)\
            .drop(data.nameDest)\
            .withColumn('type', convert_type(data.type))

        fraud = data[data.isFraud == 1]
        non_fraud = data[data.isFraud == 0]
        non_fraud = non_fraud.sample(fraction=fraud.count()/non_fraud.count())
        data = non_fraud.union(fraud)
        data = data.sample(fraction=1.0)

        data.write.parquet(str(prepared_data_path))
        return data

    def load_model(self):
        if self.model_path.exists():
            return DecisionTreeClassificationModel.load(str(self.model_path))
        return self.train_model(self.load_data())

    def _create_assembler(self):
        return VectorAssembler(inputCols=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], outputCol='features')

    def _transform_data_with_assembler(self, assembler, data, is_training_data=False):
        model_data = assembler.transform(data)
        if is_training_data:
            return model_data.select('features', data.isFraud.alias('label'))
        return model_data

    def train_model(self, data):
        decision_tree = DecisionTreeClassifier()
        model = decision_tree.fit(self._transform_data_with_assembler(self._create_assembler(), data, is_training_data=True))
        model.save(str(self.model_path))
        return model

    def cross_validate_model(self, data):
        model_data = self._transform_data_with_assembler(self._create_assembler(), data, is_training_data=True)
        decision_tree = DecisionTreeClassifier()
        evaluator = BinaryClassificationEvaluator()
        grid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=decision_tree, evaluator=evaluator, parallelism=4,
            estimatorParamMaps=grid)
        
        model = cv.fit(model_data)
        print(evaluator.evaluate(model.transform(model_data)))

    def run(self):
        pipeline_model_path = Path('data/pipeline.parquet')
        if pipeline_model_path.exists():
            pipeline_model = PipelineModel.load(str(pipeline_model_path))
        else:
            model = self.load_model()
            assembler = self._create_assembler()
            pipeline = Pipeline(stages=[assembler, model])
            pipeline_model = pipeline.fit(self.load_data())
            pipeline_model.write().save(str(pipeline_model_path))

        events = self.spark.readStream.csv('stream', header=True, schema=self.events_schema)
        predictions = pipeline_model.transform(events)
        fields = []
        for field in self.events_schema:
            fields.append(field.name)
        query = predictions.select(*fields, 'prediction').writeStream.outputMode('append')\
            .format('csv')\
            .option('path', 'results')\
            .option('checkpointLocation', 'checkpoints')\
            .start()
        query.awaitTermination()

if __name__ == '__main__':
    detector = Detector()
    detector.run()
    detector.stop()

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

public class NaiveBayesLearning {

	public static void main(String[] args) {

		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("Naive Bayes");
		JavaSparkContext sc = new JavaSparkContext(conf);
		System.out.println("Starting Training..");

		JavaRDD<LabeledPoint> training = sc.textFile(args[0]).persist(
				StorageLevel.MEMORY_ONLY_SER()).map(new Function<String, LabeledPoint>() {
			
			 // Call out Serializer in SlavesConf so need a long for serializing
			private static final long serialVersionUID = 1L;

			@Override
			public LabeledPoint call(String v1) throws Exception {
                                // Grab the label which is first number in line
				double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
				// Ignore the first two splits as first is label and second is image name
				String[] featureString = v1.split(",")[2].split(" ");
				double[] v = new double[featureString.length];
				int i = 0;
				for (String s : featureString) {
					if (s.trim().equals(""))
						continue;
					v[i++] = Double.parseDouble(s.trim());
				}
				return new LabeledPoint(label, Vectors.dense(v));
			}

		});
		
		//This function is identical to the one above, but we are creating a test data RDD instead of a training data RDD
		JavaRDD<LabeledPoint> test = sc.textFile(args[1]).persist(
				StorageLevel.MEMORY_ONLY_SER()).map(new Function<String, LabeledPoint>() {
			// Call out Serializer in SlavesConf so need a long for serializing
			private static final long serialVersionUID = 1L;

			@Override
			public LabeledPoint call(String v1) throws Exception {
				// Grab the label which is first number in line
				double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
				// Ignore the first two splits as first is label and second is image name
				String[] featureString = v1.split(",")[2].split(" ");
				double[] v = new double[featureString.length];
				int i = 0;
				for (String s : featureString) {
					// Null so continue to next
					if (s.trim().equals(""))
						continue;
					v[i++] = Double.parseDouble(s.trim());
				}
				// Return LabeledPoint with label from line and DenseVector
				return new LabeledPoint(label, Vectors.dense(v));
			}

		});

		NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

		JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
    			@Override
    			public Tuple2<Double, Double> call(LabeledPoint p) {
      				return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
    			}
  		});
		// Namenode to save at
		predictionAndLabel.coalesce(1).saveAsTextFile("hdfs://madison.cs.colostate.edu:20000/output");

		sc.close();
	}
}

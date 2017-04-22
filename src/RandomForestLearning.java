import java.util.HashMap;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.configuration.Strategy;
import org.apache.spark.mllib.tree.impurity.Gini;
import org.apache.spark.mllib.tree.impurity.Impurity;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.mllib.tree.RandomForest;

import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConverters;
import scala.collection.immutable.Map;
import scala.collection.immutable.Map.Map1;


public class RandomForestLearning {
	public static void main(String[] args) {
		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("Random Forest");
		JavaSparkContext sc = new JavaSparkContext(conf);

		/*Creating an RDD of labeled points as input data for a ML algorithm
		 * 
		 * We define an anonymous function "call" that returns a labeled point object
		 * This function is mapped across all input files, ultimately creating an RDD of all the labelled data for every image*/
		JavaRDD<LabeledPoint> training = sc.textFile(args[0]).persist(
				StorageLevel.MEMORY_ONLY_SER()).map(new Function<String, LabeledPoint>() {
			
			//I'm not entirely sure what this is for, but Eclipse complains with warnings if it's not here
			// 1L seems to be the default value
			private static final long serialVersionUID = 1L;

			@Override
			public LabeledPoint call(String v1) throws Exception {
				double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
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

			private static final long serialVersionUID = 1L;

			@Override
			public LabeledPoint call(String v1) throws Exception {
				double label = Double.parseDouble(v1.substring(0, v1.indexOf(",")));
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
		
		//Most of these values were used for the commented out call to RandomForest.trainClassifier below. I was receiving an
		//error saying that I can't cast a HashMap to an immutable Map, so I am trying a different method until I can resolve the issue
		int numClasses = 10;
		int numTrees = 10;
		String featureSubsetStrategy = "auto";
		String impurity = "gini";
		int maxDepth = 20;
		int maxBins = 34;
		int seed = 12345;
		HashMap<Object, Object> categoricalFeaturesInfo = new HashMap<>();
		Map <Object, Object> myMap = toScalaMap(categoricalFeaturesInfo);
//		scala.collection.immutable.Map<Object,Object> categoricalFeaturesInfo = new scala.collection.immutable.Map.Map1(1, 1);
		//Create a Random Forest model
//		final RandomForestModel model =
//		RandomForest.trainClassifier(training.rdd(),
//		Strategy.defaultStrategy("Classification"), numTrees,
//		featureSubsetStrategy, seed);
		 
		 
		final RandomForestModel model = RandomForest.trainClassifier(training.rdd(), numClasses,
				myMap, numTrees, featureSubsetStrategy, impurity, maxDepth,
				maxBins, seed);
		
		//This function creates an RDD of pairs in which the first value is the predicted class for an image, 
		// and the second value is the actual class of that image
		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					private static final long serialVersionUID = 1L;

					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
					}
				});
		
		predictionAndLabel.coalesce(1).saveAsTextFile("hdfs://denver:43401/output");
		sc.close();
	}
	public static <A, B> Map<A, B> toScalaMap(HashMap<A, B> m) {
	    return JavaConverters.mapAsScalaMapConverter(m).asScala().toMap(
	      Predef.<Tuple2<A, B>>conforms()
	    );
	  }
}

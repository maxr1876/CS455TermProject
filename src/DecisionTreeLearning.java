import java.util.HashMap;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;
import scala.collection.immutable.Map;

public class DecisionTreeLearning {
	public static void main(String[] args) {
		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("Decision Tree");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaRDD<LabeledPoint> training = sc.textFile(args[0]).persist(
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
		
		Integer numClasses = 10;
		HashMap<Object, Object> categoricalFeaturesInfo = new HashMap<>();
		Map <Object, Object> myMap = RandomForestLearning.toScalaMap(categoricalFeaturesInfo);
		String impurity = "gini";
		Integer maxDepth = 5;
		Integer maxBins = 32;

		final org.apache.spark.mllib.tree.model.DecisionTreeModel model = DecisionTree.trainClassifier(training.rdd(), numClasses,
				  myMap, impurity, maxDepth, maxBins);
		
		JavaPairRDD<Double, Double> predictionAndLabel =
			test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
				private static final long serialVersionUID = 1L;

				@Override
				public Tuple2<Double, Double> call(LabeledPoint p) {
					return new Tuple2<>(model.predict(p.features()), p.label());
			  	}
			});				
		
		predictionAndLabel.coalesce(1).saveAsTextFile("hdfs://denver:43401/DTOutput/"+new Random().nextInt(1000000000));
		sc.close();
	}
}

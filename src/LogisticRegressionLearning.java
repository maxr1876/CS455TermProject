import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;

public class LogisticRegressionLearning {
	public static void main(String[] args){
		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("Logistic Regression");
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
		
		final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
				.setNumClasses(10)
				.run(training.rdd());
		
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
				new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double prediction = model.predict(p.features());
						return new Tuple2<Object, Object>(prediction, p.label());
					}
				});
				
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		System.out.println("Accuracy = " + accuracy);
				
		predictionAndLabels.coalesce(1).saveAsTextFile("hdfs://nashville.cs.colostate.edu:30101/output");
		sc.close();
	}
}

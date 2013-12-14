package WekaCode;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
//import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Random;

public class StartWeka {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("/home/tanay/Copy/Data/Output/hay-train.arff"));
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes()-1);
		breader.close();
		NaiveBayes nB = new NaiveBayes();
		nB.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(nB, train, 10, new Random(1));
		System.out.println(eval.toSummaryString("\nResults\n=====\n",true));
		System.out.println("F-Score :"+eval.fMeasure(1)+"\nPrecision :"+eval.precision(1)+"\nRecall :"+eval.recall(1));

		
		 AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		  CfsSubsetEval eval1 = new CfsSubsetEval();
		  GreedyStepwise search = new GreedyStepwise();
		  search.setSearchBackwards(true);
		  J48 base = new J48();
		  //NaiveBayes base = new NaiveBayes();
		  classifier.setClassifier(base);
		  classifier.setEvaluator(eval1);
		  classifier.setSearch(search);
		  // 10-fold cross-validation
		  Evaluation evaluation = new Evaluation(train);
		  //evaluation.evaluateModel(classifier, train, 10);
		  evaluation.crossValidateModel(classifier, train, 10, new Random(1));
		  System.out.println(evaluation.toSummaryString());
		  System.out.println("F-Score :"+evaluation.fMeasure(1)+"\nPrecision :"+evaluation.precision(1)+"\nRecall :"+evaluation.recall(1));
	}

}

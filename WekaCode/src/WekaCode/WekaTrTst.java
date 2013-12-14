package WekaCode;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
//import weka.classifiers.Classifier;
/////////////////////////////////////Classifiers/////////////////////////////////////////////
//import weka.classifiers.lazy.IBk;
//import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.bayes.BayesNet;
//import weka.classifiers.trees.LMT;
//import weka.classifiers.trees.RandomForest;
//import weka.classifiers.trees.DecisionStump;
//import weka.classifiers.trees.REPTree;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.LibSVM;
///////////////////////////////////////////////////////////////////////////////////////////
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.evaluation.Evaluation;


import java.io.File;
//import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.text.DecimalFormat;

//import java.io.FileWriter;
//import java.util.Random;
//import wekaexamples.core.converters.*;

public class WekaTrTst {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("/home/tanay/Copy/Data/Output/hay-train.arff"));
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes() -1);
		
		breader = new BufferedReader(new FileReader("/home/tanay/Copy/Data/Output/hay-test.arff"));
		Instances test = new Instances(breader);
		test.setClassIndex(train.numAttributes() -1);
		
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		AttributeSelectedClassifier classifier1 = new AttributeSelectedClassifier();
		CfsSubsetEval eval1 = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		
		////////////////////////////////////////Classifiers//////////////////////////////////////////
		
		AdaBoostM1 abase = new AdaBoostM1();
		J48 jbase = new J48();
		LMT lbase = new LMT();
		RandomForest rbase = new RandomForest();
		IBk ibase = new IBk();
		LibSVM svm = new LibSVM();
		
		DecisionStump dbase = new DecisionStump();
		KStar kbase = new KStar();		
		REPTree Rbase = new REPTree();
		
		/////////////////////////Modify parameters of base classifier///////////////////////////////////
		
		
		////////////////////////////////Set the base classifier////////////////////////////////////////
		  
		abase.setClassifier(lbase);
		
		//////////////////////////////////////////////////////////////////////////////////////////
		
		
		classifier.setClassifier(abase);
		classifier.setEvaluator(eval1);
		classifier.setSearch(search);
		  
				
		//abase.buildClassifier(train);		
		//jbase.buildClassifier(train);
		
		//////////////////////////////////////////////////////////////
		DecimalFormat df = new DecimalFormat("#.###");
		//////////////////////////////////////////////////////////////
		
		classifier.buildClassifier(train);
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(classifier, test);
		//eval.crossValidateModel(tree, train, 10, new Random(1));
		System.out.println(eval.toSummaryString("\nResults\n=====\n",true));
		System.out.println("\n//////////////////////////Result of Adaboost program/////////////////////////////////");
		System.out.println("F-Score   :"+df.format(eval.fMeasure(1))+"\nPrecision :"+df.format(eval.precision(1))+"\nRecall     :"+df.format(eval.recall(1)));
		
	
		TrAdaBoost tab = new TrAdaBoost(train); 
		
		/////////////////////////////////////////////
		tab.setClassifier(lbase);
		//classifier1.setClassifier(tab);
		
		
		//////////////////////////////////////////////
		
		//classifier1.setClassifier(rbase);
		tab.buildClassifier(train);
		Evaluation eval2 = new Evaluation(train);
		eval2.evaluateModel(tab, test);
		System.out.println("\n//////////////////////////Result of TrAdaboost program/////////////////////////////////");
		System.out.println("F-Score   :"+df.format(eval2.fMeasure(1))+"\nPrecision :"+df.format(eval2.precision(1))+"\nRecall     :"+df.format(eval2.recall(1)));

		
		 
	    //    svm.buildClassifier(train);
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		/*for(int t=0;t<4;t++)
			System.out.println("F-Score   :"+t+":"+eval.fMeasure(t));*/
		
		/*Instances labeled = new Instances(test);
		
		for(int i=0;i<test.numInstances();i++)
		{
			double clsLabel = tree.classifyInstance(test.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
		}
			BufferedWriter writer=new BufferedWriter(new FileWriter("/home/tanay/Copy/Data/Output/Label.arff"));
			writer.write(labeled.toString());*/
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		breader.close();
	}

}

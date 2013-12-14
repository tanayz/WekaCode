package WekaCode;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.text.DecimalFormat;
import java.util.Random;






////////////////////////////////////////////////////////
//import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.SimpleCart;

//////////////////////////////////////////////////////
import milk.classifiers.MDD;
//////////////////////////////////////////////////////
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class WekaCSV {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		
			
   
	    CSVLoader loader = new CSVLoader();
	    loader.setSource(new File("/home/tanay/Copy/Data/Output/Final_250_t2010.csv"));
        Instances data = loader.getDataSet();
        loader.setSource(new File("/home/tanay/Copy/Data/Output/Final_250_t2011.csv"));
        Instances data1 = loader.getDataSet();

	    System.out.println("\nHeader of dataset:\n");
	    //System.out.println(new Instances(data));
	    
	    DecimalFormat df = new DecimalFormat("#.###");
	    
	 // save ARFF
        ArffSaver saver = new ArffSaver();
        ArffSaver saver1 = new ArffSaver();
        saver.setInstances(data);
        saver1.setInstances(data1);
        saver.setFile(new File("/home/tanay/Copy/Data/Output/train.arff"));
        saver1.setFile(new File("/home/tanay/Copy/Data/Output/test.arff"));
        saver.setDestination(new File("/home/tanay/Copy/Data/Output/train.arff"));
        saver1.setDestination(new File("/home/tanay/Copy/Data/Output/test.arff"));
        saver.writeBatch();
        saver1.writeBatch();

        BufferedReader br=null,br1=null;
        br=new BufferedReader(new FileReader("/home/tanay/Copy/Data/Output/train.arff"));
        br1=new BufferedReader(new FileReader("/home/tanay/Copy/Data/Output/test.arff"));
        Instances train=new Instances(br);
        Instances test=new Instances(br1);
        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(train.numAttributes()-1);
        br.close();br1.close();
        
        ClassificationViaRegression cb = new ClassificationViaRegression();
        InputMappedClassifier ib = new InputMappedClassifier();
        AdaBoostM1 ab = new AdaBoostM1();
        J48 jb = new J48();
        RandomForest rb = new RandomForest();
         LMT lb = new LMT();
         M5P mb = new M5P();
        SimpleCart sb=new SimpleCart();
        
        ///////////////////////Multiple Instance//////////////////////////////////////////////
        
        
        
        ///////////////////////////////////////////////////////////////////////////////////
         rb.setNumTrees(60);
         TrAdaBoost tab = new TrAdaBoost(train);
         
        //  System.out.println("jb\n"+jb.getBinarySplits()+"\n"+jb.getCollapseTree()+"\n"+jb.getConfidenceFactor()+"\n"+jb.getDebug()
        //		+"\n"+jb.getMinNumObj()+"\n"+jb.getNumFolds()+"\n"+jb.getReducedErrorPruning()+"\n"+jb.getSaveInstanceData()
        //	+"\n"+jb.getSeed()+"\n"+jb.getSubtreeRaising()+"\n"+jb.getUnpruned()+"\n"+jb.getUseLaplace()+"\n"+jb.getUseMDLcorrection());
     
        
        // System.out.println("ib\n"+ib.getIgnoreCaseForNames()+"\n"+ib.getSuppressMappingReport()+"\n"+ib.getTrim());
        
        ab.setClassifier(sb);
        ib.setClassifier(ab);
        ib.buildClassifier(train);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(ib, test);
        //eval.crossValidateModel(jb, train, 9, new Random(1));
        System.out.println(eval.toSummaryString("\nResults\n=====\n",true));
        System.out.println("\n//////////////////////////Result of program/////////////////////////////////");
       
        System.out.println("F-Score   :"+df.format(eval.fMeasure(1))+"\nPrecision :"+df.format(eval.precision(1))
        		+"\nRecall     :"+df.format(eval.recall(1)));


	}

}

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MIWrapper.java
 *    Copyright (C) 2002 Eibe Frank, Xin Xu
 *
 */
package milk.classifiers;
import milk.core.*;
import weka.classifiers.*;
import java.util.*;
import java.io.*;
import weka.core.*;

/**
 * 
 * Weighted Wrapper method from Eibe 
 *
 * Valid options are:<p>
 *
 * -D <br>
 * Turn on debugging output.<p>
 *
 * -W classname <br>
 * Specify the full class name of a classifier as the basis (required).<p>
 *
 * -P method index <br>
 * Set which method to use in testing: 1.arithmatic average; 2.geometric average. (default: 1)<p>
 *
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.13 $ 
 */
public class MIWrapper 
    extends MIClassifier 
    implements OptionHandler, MITransform {  
    
    /** The index of the class attribute */
    protected int m_ClassIndex;

    /** The number of the class labels */
    protected int m_NumClasses;
    protected int m_IdIndex;    
    
    /** Debugging output */
    protected boolean m_Debug;
   
    /** All attribute names */
    protected Instances m_Attributes;

    protected Classifier m_Classifier = new weka.classifiers.rules.ZeroR();   
    
    protected int m_Method = 1;

    //protected double[] m_Prior=null;
    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {
	
	Vector newVector = new Vector(3);	
	newVector.addElement(new Option("\tTurn on debugging output.",
					"D", 0, "-D"));
	newVector.addElement(new Option("\tThe method used in testing:\n"+
					"\t1.arithmatic average; 2.geometric average\n"+
					"\t(default: 1)",
					"P", 1, "-P <num>"));
	if ((m_Classifier != null) &&
	    (m_Classifier instanceof OptionHandler)) {
	    newVector.addElement(new Option("",
					    "", 0, "\nOptions specific to classifier "
					    + m_Classifier.getClass().getName() + ":"));
	    Enumeration enum = ((OptionHandler)m_Classifier).listOptions();
	    while (enum.hasMoreElements()) {
		newVector.addElement(enum.nextElement());
	    }
	}
	return newVector.elements();
    }
    
    
    /**
     * Parses a given list of options. Valid options are:<p>
     *
     * -D <br>
     * Turn on debugging output.<p>
     *
     * -W classname <br>
     * Specify the full class name of a classifier as the basis (required).<p>
     *
     * -P method_index <br>
     * Set which method to use in testing: 1.arithmatic average; 2.geometric average. (default: 1)<p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
	
	setDebug(Utils.getFlag('D', options));
	
	String methodString = Utils.getOption('P', options);
	if (methodString.length() != 0) {
	    setMethod(Integer.parseInt(methodString));
	} else {
	    setMethod(1);
	}
	
	String classifierName = Utils.getOption('W', options);
	if (classifierName.length() == 0) {
	    throw new Exception("A classifier must be specified with"
				+ " the -W option.");
	}
	setClassifier(Classifier.forName(classifierName,
					 Utils.partitionOptions(options)));
    }
    
    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
	
	String [] classifierOptions = new String [0];
	if ((m_Classifier != null) && 
	    (m_Classifier instanceof OptionHandler)) {
	    classifierOptions = ((OptionHandler)m_Classifier).getOptions();
	}
	
	String [] options = new String [classifierOptions.length + 6];
	int current = 0;
	if (getDebug()) {
	    options[current++] = "-D";
	}
	
	options[current++] = "-P"; 
	options[current++] = "" + getMethod();
	
	if (getClassifier() != null) {
	    options[current++] = "-W";
	    options[current++] = getClassifier().getClass().getName();
	}
	options[current++] = "--";

	System.arraycopy(classifierOptions, 0, options, current, 
			 classifierOptions.length);
	current += classifierOptions.length;
	while (current < options.length) {
	    options[current++] = "";
	}
	return options;
    }
    
    
    /**
     * Sets whether debugging output will be printed.
     *
     * @param debug true if debugging output should be printed
     */
    public void setDebug(boolean debug) {
	m_Debug = debug;
    }
    
    /**
     * Gets whether debugging output will be printed.
     *
     * @return true if debugging output will be printed
     */
    public boolean getDebug() {
	return m_Debug;
    }
      
    /**
     * Set the base classifier. 
     *
     * @param newClassifier the Classifier to use.
     */
    public void setClassifier(Classifier newClassifier) {
	
	m_Classifier = newClassifier;
    }
    
    /**
     * Get the classifier used as the classifier
     *
     * @return the classifier used as the classifier
     */
    public Classifier getClassifier() {
	
	return m_Classifier;
    }
    
    /**
     * Set the method used in testing. 
     *
     * @param newMethod the index of method to use.
     */
    public void setMethod(int newMethod) {
	
	m_Method = newMethod;
    }
    
    /**
     * Get the method used in testing.
     *
     * @return the index of method used in testing.
     */
    public int getMethod() {
	
	return m_Method;
    }

    // Implements MITransform 
    public Instances transform(Exemplars train) throws Exception{
	
	Instances data=new Instances(m_Attributes);// Data to learn a model	
	data.deleteAttributeAt(m_IdIndex);// ID attribute useless	
	Instances dataset = new Instances(data,0);
	double sumNi = 0, // Total number of instances
	    N = train.numExemplars(); // Number of exemplars
	
	for(int i=0; i<N; i++)
	    sumNi += train.exemplar(i).getInstances().numInstances();
	
	// Initialize weights
	for(int i=0; i<N; i++){
	    Exemplar exi = train.exemplar(i);
	    // m_Prior[(int)exi.classValue()]++;
	    Instances insts = exi.getInstances();
	    double ni = (double)insts.numInstances();    
	    for(int j=0; j<ni; j++){	
		Instance ins = new Instance(insts.instance(j));// Copy		
		ins.deleteAttributeAt(m_IdIndex);
		ins.setDataset(dataset);
		ins.setWeight(sumNi/(N*ni));
		//ins.setWeight(1);
		data.add(ins);
	    }
	}

	return data;
    }
    
    /**
     * Builds the classifier
     *
     * @param train the training data to be used for generating the
     * boosted classifier.
     * @exception Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Exemplars train) throws Exception {
	
	if (train.classAttribute().type() != Attribute.NOMINAL) {
	    throw new Exception("Class attribute must be nominal.");
	}
	if (train.checkForStringAttributes()) {
	    throw new Exception("Can't handle string attributes!");
	}	
	if (m_Classifier == null) {
	    throw new Exception("A base classifier has not been specified!");
	}
	/*if (! (m_Classifier instanceof DistributionClassifier)) {
	    throw new Exception("A base classifier is not a DistributionClassifier!");
	    }*/
	System.out.println("Start training ...");
	m_ClassIndex = train.classIndex();
	m_IdIndex = train.idIndex();
	m_NumClasses = train.numClasses();
	m_Attributes = new Instances(train.exemplar(0).getInstances(), 0);
	//m_Prior = new double[m_NumClasses];	
	Instances data = transform(train);
	m_Classifier.buildClassifier(data);
	//Utils.normalize(m_Prior);
    }		
    
    /**
     * Computes the distribution for a given exemplar
     *
     * @param exmp the exemplar for which distribution is computed
     * @return the distribution
     * @exception Exception if the distribution can't be computed successfully
     */
    public double[] distributionForExemplar(Exemplar exmp) 
	throws Exception {
	
	// Extract the data
	Instances insts = new Instances(exmp.getInstances());	
	double nI = (double)insts.numInstances();
	insts.deleteAttributeAt(m_IdIndex);// ID attribute useless	
	
	// Compute the log-probability of the bag
	double [] distribution = new double[m_NumClasses];
	
	for(int i=0; i<nI; i++){
	    double[] dist = m_Classifier.distributionForInstance(insts.instance(i));
	    for(int j=0; j<m_NumClasses; j++){
		
		switch(m_Method){
		case 1:
		    distribution[j] += dist[j]/nI;
		    break;
		case 2:		    
		    // Avoid 0/1 probability
		    if(dist[j]<0.001)
			dist[j] = 0.001;
		    else if(dist[j]>0.999)
			dist[j] = 0.999;
		    
		    distribution[j] += Math.log(dist[j])/nI;
		    break;
		    /*case 3:
		    distribution[j] += 
			Math.exp(Math.log(dist[j])/nI +
				 Math.log(m_Prior[j])*(nI-1.0)/nI);
		    */
		}
	    }
	}
	
	if(m_Method == 2)
	    for(int j=0; j<m_NumClasses; j++)
		distribution[j] = Math.exp(distribution[j]);
	
	Utils.normalize(distribution);
	return distribution;
    }
       
    /**
     * Gets a string describing the classifier.
     *
     * @return a string describing the classifer built.
     */
    public String toString() {	
	return "MIWrapper with base classifier: \n"+m_Classifier.toString();
    }
    
    /**
     * Main method for testing this class.
     *
     * @param argv should contain the command line arguments to the
     * scheme (see Evaluation)
     */
    public static void main(String [] argv) {
	try {
	    System.out.println(MIEvaluation.evaluateModel(new MIWrapper(), argv));
	} catch (Exception e) {
	    e.printStackTrace();
	  System.err.println(e.getMessage());
	}
    }
}

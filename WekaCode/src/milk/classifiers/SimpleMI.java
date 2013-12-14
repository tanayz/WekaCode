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
 *    SimpleMI.java
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
 * Avg of feature data and reduce MI data into mono-instance
 *
 * Valid options are:<p>
 *
 * -M method index <br>
 * Set which method to use in transformation: 1.arithmatic average; 2.geometric centor
 * for each bag. (default: 1)<p>
 *
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.13 $ 
 */
public class SimpleMI extends MIWrapper implements OptionHandler, MITransform {  
  
    protected int m_TransformMethod = 1;
    protected Exemplars m_Exemplars;
    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {
	
	Vector newVector = new Vector(1);	
	newVector.addElement(new Option("\tThe method used in transformation:\n"+
					"\t1.arithmatic average; 2.geometric centor\n"+
					"\tof a bag (default: 1)",
					"M", 1, "-M <num>"));
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
     * -M method_index <br>
     * Set which method to use in transformation: 1.arithmatic average; 2.geometric centor
     * of a bag (default: 1)<p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {	
	
	String methodString = Utils.getOption('M', options);
	if (methodString.length() != 0) {
	    setTransformMethod(Integer.parseInt(methodString));
	} else {
	    setTransformMethod(1);
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
	
	String [] options = new String [classifierOptions.length + 5];
	int current = 0;
	options[current++] = "-M"; 
	options[current++] = "" + getTransformMethod();
	
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
     * Set the method used in transformation. 
     *
     * @param newMethod the index of method to use.
     */
    public void setTransformMethod(int newMethod) {
	
	m_TransformMethod = newMethod;
    }
    
    /**
     * Get the method used in transformation.
     *
     * @return the index of method used.
     */
    public int getTransformMethod() {
	
	return m_TransformMethod;
    }

    // Implements MITransform 
    public Instances transform(Exemplars train) throws Exception{
	
	Instances data=new Instances(m_Attributes);// Data to learn a model	
	data.deleteAttributeAt(m_IdIndex);// ID attribute useless	
	Instances dataset = new Instances(data,0);
	Instance template = new Instance(dataset.numAttributes());
	template.setDataset(dataset);
	double N = train.numExemplars(); // Number of exemplars
		
	for(int i=0; i<N; i++){
	    Exemplar exi = train.exemplar(i);
	    Instances insts = exi.getInstances();
	    int attIdx = 0;
	    Instance newIns = new Instance(template);
	    newIns.setDataset(dataset);
	    for(int j=0; j<insts.numAttributes(); j++){	
		if((j==m_IdIndex) || (j==m_ClassIndex))
		    continue;
		double value;
		if(m_TransformMethod==1){
		    value = insts.meanOrMode(j);
		}
		else{
		    double[] minimax = minimax(insts, j);
		    value = (minimax[0]+minimax[1])/2.0;
		}
		newIns.setValue(attIdx++, value);
	    }
	    newIns.setClassValue(exi.classValue());
	    data.add(newIns);
	}
	
	return data;
    }
    
    /**
     * Get the minimal and maximal value of a certain attribute in a certain data
     *
     * @param data the data
     * @param attIndex the index of the attribute
     * @return the double array containing in entry 0 for min and 1 for max.
     */
    public static double[] minimax(Instances data, int attIndex){
	double[] rt = {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
	for(int i=0; i<data.numInstances(); i++){
	    double val = data.instance(i).value(attIndex);
	    if(val > rt[1])
		rt[1] = val;
	    if(val < rt[0])
		rt[0] = val;
	}
	
	for(int j=0; j<2; j++)
	    if(Double.isInfinite(rt[j]))
		rt[j] = Double.NaN;
	
	return rt;
    }

    /**
     * Builds the classifier
     *
     * @param train the training data to be used for generating the
     * boosted classifier.
     * @exception Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Exemplars train) throws Exception {

	super.buildClassifier(train);
	m_Exemplars = new Exemplars(train, 1);
    }		
    
    /**
     * Computes the distribution for a given exemplar
     *
     * @param exmp the exemplar for which distribution is computed
     * @return the distribution
     * @exception Exception if the distribution can't be computed successfully
     */
    //public double[] distributionForExemplar(Exemplar exmp)
    public double classifyExemplar(Exemplar exmp) 
	throws Exception {
	Exemplars test = new Exemplars(m_Exemplars, 1);
	test.add(exmp);
	Instance datum = transform(test).firstInstance();
	//return ((DistributionClassifier)m_Classifier).
	//  distributionForInstance(datum);	
	return m_Classifier.classifyInstance(datum);	   
    }
    
    /**
     * Gets a string describing the classifier.
     *
     * @return a string describing the classifer built.
     */
    public String toString() {	
	return "SimpleMI with base classifier: \n"+m_Classifier.toString();
    }
    
    /**
     * Main method for testing this class.
     *
     * @param argv should contain the command line arguments to the
     * scheme (see Evaluation)
     */
    public static void main(String [] argv) {
	try {
	    System.out.println(MIEvaluation.evaluateModel(new SimpleMI(), argv));
	} catch (Exception e) {
	    e.printStackTrace();
	  System.err.println(e.getMessage());
	}
    }
}

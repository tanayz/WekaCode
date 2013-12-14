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
 *    MILRGEOM.java
 *    Copyright (C) 2002 Eibe Frank, Xin Xu
 *
 */
package milk.classifiers;
import milk.core.*;
import weka.classifiers.*;
import java.util.*;
import java.io.*;
import weka.core.*;
import weka.core.Matrix;
import weka.filters.*;

/**
 * 
 * Normalized goemetric mean is taken on the posteriors of instances, 
 * regardless of class labels
 *
 * Valid options are:<p>
 *
 * -D <br>
 * Turn on debugging output.<p>
 *
 * -P precision <br>
 * Set the precision of stopping criteria in Newton method.<p>
 *
 * -R ridge <br>
 * Set the ridge parameter for the log-likelihood.<p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.13 $ 
 */
public class MILRGEOM extends MIClassifier implements OptionHandler {
  
    
    /** The index of the class attribute */
    protected int m_ClassIndex;
    
    protected double[] m_Par;

    /** The number of the class labels */
    protected int m_NumClasses;
    protected int m_IdIndex;    
    /** The ridge parameter. */
    protected double m_Ridge = 1e-6;
    
    /** The filter used to make attributes numeric. */
    //private NominalToBinaryFilter m_NominalToBinary;
    
    /** The filter used to get rid of missing values. */
    //private ReplaceMissingValuesFilter m_ReplaceMissingValues;
    
    /** Debugging output */
    protected boolean m_Debug;
   
    /** Class labels for each bag */
    protected int[] m_Classes;

    /** MI data */ 
    protected double[][][] m_Data;

    /** All attribute names */
    protected Instances m_Attributes;

    protected double[] xMean=null, xSD=null;
    
    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(3);
	newVector.addElement(new Option("\tTurn on debugging output.",
					"D", 0, "-D"));
	newVector.addElement(new Option("\tSet the ridge in the log-likelihood.",
					"R", 1, "-R <ridge>"));
	return newVector.elements();
    }
    
    /**
     * Parses a given list of options. Valid options are:<p>
     *
     * -D <br>
     * Turn on debugging output.<p>
     *
     * -P precision <br>
     * Set the precision of stopping criteria in Newton method.<p>
     *
     * -R ridge <br>
     * Set the ridge parameter for the log-likelihood.<p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
	setDebug(Utils.getFlag('D', options));
	
	String ridgeString = Utils.getOption('R', options);
	if (ridgeString.length() != 0) 
	    m_Ridge = Double.parseDouble(ridgeString);
	else 
	    m_Ridge = 1.0e-6;
    }
    
    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
	
	String [] options = new String [3];
	int current = 0;
	
	if (getDebug()) {
	    options[current++] = "-D";
	}
	options[current++] = "-R";
	options[current++] = ""+m_Ridge;
	
	while (current < options.length) 
	    options[current++] = "";
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
     * Sets the ridge in the log-likelihood.
     *
     * @param ridge the ridge
     */
    public void setRidge(double ridge) {
	m_Ridge = ridge;
    }
    
    /**
     * Gets the ridge in the log-likelihood.
     *
     * @return the ridge
     */
    public double getRidge() {
	return m_Ridge;
    }
    
    private class OptEng extends Optimization{
	/** 
	 * Evaluate objective function
	 * @param x the current values of variables
	 * @return the value of the objective function 
	 */
	protected double objectiveFunction(double[] x){
	    double nll = 0; // -LogLikelihood
	    for(int i=0; i<m_Classes.length; i++){ // ith bag
		int nI = m_Data[i][0].length; // numInstances in ith bag
		double bag = 0;   // Log-prob. 

		for(int j=0; j<nI; j++){
		    double exp=0.0;
		    for(int k=m_Data[i].length-1; k>=0; k--)
			exp += m_Data[i][k][j]*x[k+1];
		    exp += x[0];
		    
		    if(m_Classes[i]==1)
			bag -= exp/(double)nI;
		    else
			bag += exp/(double)nI;
		}
		
		nll += Math.log(1.0+Math.exp(bag));
	    }		
	    
	    // ridge: note that intercepts NOT included
	    for(int r=1; r<x.length; r++)
		nll += m_Ridge*x[r]*x[r];
	    
	    return nll;
	}
	
	/** 
	 * Evaluate Jacobian vector
	 * @param x the current values of variables
	 * @return the gradient vector 
	 */
	protected double[] evaluateGradient(double[] x){
	    double[] grad = new double[x.length];
	    for(int i=0; i<m_Classes.length; i++){ // ith bag
		int nI = m_Data[i][0].length; // numInstances in ith bag		
		double bag = 0;
		double[] sumX = new double[x.length];
		for(int j=0; j<nI; j++){
		    // Compute exp(b0+b1*Xi1j+...)/[1+exp(b0+b1*Xi1j+...)]
		    double exp=0.0;
		    for(int k=m_Data[i].length-1; k>=0; k--)
			exp += m_Data[i][k][j]*x[k+1];
		    exp += x[0];
		
		    if(m_Classes[i]==1){
			bag -= exp/(double)nI;
			for(int q=0; q<grad.length; q++){
			    double m = 1.0;
			    if(q>0) m=m_Data[i][q-1][j];
			    sumX[q] -= m/(double)nI;
			}
		    }
		    else{
			bag += exp/(double)nI;
			for(int q=0; q<grad.length; q++){
			    double m = 1.0;
			    if(q>0) m=m_Data[i][q-1][j];
			    sumX[q] += m/(double)nI;
			}			
		    }
		}
		
		for(int p=0; p<x.length; p++)
		    grad[p] += Math.exp(bag)*sumX[p]/(1.0+Math.exp(bag));
	    }
	    // ridge: note that intercepts NOT included
	    for(int r=1; r<x.length; r++){
		grad[r] += 2.0*m_Ridge*x[r];
	    }
	    
	    return grad;
	}
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
	/*train = new Instances(train);
	train.deleteWithMissingClass();
	if (train.numInstances() == 0) {
	    throw new Exception("No train instances without missing class value!");
	}
	m_ReplaceMissingValues = new ReplaceMissingValuesFilter();
	m_ReplaceMissingValues.setInputFormat(train);
	train = Filter.useFilter(train, m_ReplaceMissingValues);
	m_NominalToBinary = new NominalToBinaryFilter();
	m_NominalToBinary.setInputFormat(train);
	train = Filter.useFilter(train, m_NominalToBinary);*/
	m_ClassIndex = train.classIndex();
	m_IdIndex = train.idIndex();
	m_NumClasses = train.numClasses();
	
	int nK = 1;                     // Only K-1 class labels needed 
	int nR = train.numAttributes() - 2;
	int nC = train.numExemplars();
	
	m_Data  = new double [nC][nR][];              // Data values
	m_Classes  = new int [nC];                    // Class values
	m_Attributes = new Instances(train.exemplar(0).getInstances(),0);	

	xMean= new double [nR];             // Mean of mean
	xSD  = new double [nR];             // Mode of stddev
	int g1NE = 0;                                 // # of bags with >1 instances
	double sY1=0, sY0=0, totIns=0;                          // Number of classes
	int[] missingbags = new int[nR];

	if (m_Debug) {
	    System.out.println("Extracting data...");
	}
	
	for(int h=0; h<m_Data.length; h++){
	    Exemplar current = train.exemplar(h);
	    m_Classes[h] = (int)current.classValue();  // Class value starts from 0
	    Instances currInsts = current.getInstances();
	    int nI = currInsts.numInstances();
	    totIns += (double)nI;

	    int idx=0;
	    for (int i = 0; i < train.numAttributes(); i++) {  		
		if ((i==m_ClassIndex) || (i==m_IdIndex))
		    continue;
		
		// initialize m_data[][][]		
		m_Data[h][idx] = new double[nI];
		double avg=0, std=0, num=0;
		for (int k=0; k<nI; k++){
		    if(!currInsts.instance(k).isMissing(i)){
			m_Data[h][idx][k] = currInsts.instance(k).value(i);	 	  
			//xMean[idx] += m_Data[h][idx][k];
			//xSD[idx] += m_Data[h][idx][k]*m_Data[h][idx][k];
			avg += m_Data[h][idx][k];
			std += m_Data[h][idx][k]*m_Data[h][idx][k];
			num++;
		    }
		    else
			m_Data[h][idx][k] = Double.NaN;
		}
		if(num > 0){
		    xMean[idx] += avg/num;
		    xSD[idx] += std/num;
		}
		else
		    missingbags[idx]++;		
		idx++;
	    }	    
	    
	    // Class count	
	    if (m_Classes[h] == 1)
		sY1++;
	    else
		sY0++;
	}
	
	for (int j = 0; j < nR; j++) {
	    //xMean[j] = xMean[j]/totIns;
	    //xSD[j] = Math.sqrt(xSD[j]/(totIns-1.0)-xMean[j]*xMean[j]*totIns/(totIns-1.0));
	    xMean[j] = xMean[j]/(double)(nC-missingbags[j]);
	    xSD[j] = Math.sqrt(xSD[j]/((double)(nC-missingbags[j])-1.0)
	    		       -xMean[j]*xMean[j]*(double)(nC-missingbags[j])/
			       ((double)(nC-missingbags[j])-1.0));	
	}
	
	if (m_Debug) {	    
	    // Output stats about input data
	    System.out.println("Descriptives...");
	    System.out.println(sY0 + " bags have class 0 and " +
			       sY1 + " bags have class 1");
	    System.out.println("\n Variable     Avg       SD    ");
	    for (int j = 0; j < nR; j++) 
		System.out.println(Utils.doubleToString(j,8,4) 
				   + Utils.doubleToString(xMean[j], 10, 4) 
				   + Utils.doubleToString(xSD[j], 10,4));
	}
	
	// Normalise input data and remove ignored attributes
	for (int i = 0; i < nC; i++) {
	    for (int j = 0; j < nR; j++) {
		for(int k=0; k < m_Data[i][j].length; k++){
		    if(xSD[j] != 0){
			if(!Double.isNaN(m_Data[i][j][k]))
			    m_Data[i][j][k] = (m_Data[i][j][k] - xMean[j]) / xSD[j];
			else
			    m_Data[i][j][k] = 0;
		    }
		}
	    }
	}
	
	if (m_Debug) {
	    System.out.println("\nIteration History..." );
	}
	
	double x[] = new double[nR + 1];
	x[0] =  Math.log((sY1+1.0) / (sY0+1.0));
	double[][] b = new double[2][x.length];
	b[0][0] = Double.NaN;
	b[1][0] = Double.NaN;
	for (int q=1; q < x.length;q++){
	    x[q] = 0.0;		
	    b[0][q] = Double.NaN;
	    b[1][q] = Double.NaN;
	}
	
	OptEng opt = new OptEng();	
	opt.setDebug(m_Debug);
	m_Par = opt.findArgmin(x, b);
	while(m_Par==null){
	    m_Par = opt.getVarbValues();
	    if (m_Debug)
		System.out.println("200 iterations finished, not enough!");
	    m_Par = opt.findArgmin(m_Par, b);
	}
	if (m_Debug)
	    System.out.println(" -------------<Converged>--------------");
	
	// Convert coefficients back to non-normalized attribute units
	for(int j = 1; j < nR+1; j++) {
	    if (xSD[j-1] != 0) {
		m_Par[j] /= xSD[j-1];
		m_Par[0] -= m_Par[j] * xMean[j-1];
	    }
	}
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
	
	/*m_ReplaceMissingValues.input(instance);
	instance = m_ReplaceMissingValues.output();
	m_NominalToBinary.input(instance);
	instance = m_NominalToBinary.output();
	*/

	// Extract the data
	Instances ins = exmp.getInstances();
	int nI = ins.numInstances(), nA = ins.numAttributes();
	double[][] dat = new double [nI][nA+1-2];
	for(int j=0; j<nI; j++){
	    dat[j][0]=1.0;
	    int idx=1;
	    for(int k=0; k<nA; k++){ 
		if((k==m_ClassIndex) || (k==m_IdIndex))
		    continue;
		if(!ins.instance(j).isMissing(k))
		    dat[j][idx] = ins.instance(j).value(k);
		else
		    dat[j][idx] = xMean[idx-1];
		idx++;
	    }
	}
	
	// Compute the log-odds of the bag
	double [] distribution = new double[m_NumClasses];
	
	for(int i=0; i<nI; i++){
	    double exp = 0.0;
	    for(int r=0; r<m_Par.length; r++)
		exp += m_Par[r]*dat[i][r];
	    distribution[1] += exp/(double)nI; 
	}
	
	distribution[1] = 1.0/(1.0+Math.exp(-distribution[1]));
	distribution[0] = 1-distribution[1];
	
	//Utils.normalize(distribution);
	return distribution;
    }
    
    /**
     * Gets a string describing the classifier.
     *
     * @return a string describing the classifer built.
     */
    public String toString() {
	
	//double CSq = m_LLn - m_LL;
	//int df = m_NumPredictors;
	String result = "Modified Logistic Regression";
	if (m_Par == null) {
	    return result + ": No model built yet.";
	}
	/*    result += "\n\nOverall Model Fit...\n" 
	      +"  Chi Square=" + Utils.doubleToString(CSq, 10, 4) 
	      + ";  df=" + df
	      + ";  p=" 
	      + Utils.doubleToString(Statistics.chiSquaredProbability(CSq, df), 10, 2)
	      + "\n";
	*/
	
	result += "\nCoefficients...\n"
	    + "Variable      Coeff.\n";
	for (int j = 1, idx=0; j < m_Par.length; j++, idx++) {
	    if((idx == m_ClassIndex) || (idx==m_IdIndex))
		idx++;
	    result += m_Attributes.attribute(idx).name();
	    result += " "+Utils.doubleToString(m_Par[j], 12, 4); 
	    result += "\n";
	}
	
	result += "Intercept:";
	result += " "+Utils.doubleToString(m_Par[0], 10, 4); 
	result += "\n";
	
	result += "\nOdds Ratios...\n"
	    + "Variable         O.R.\n";
	for (int j = 1, idx=0; j < m_Par.length; j++, idx++) {
	    if((idx == m_ClassIndex) || (idx==m_IdIndex))
		idx++;
	    result += " " + m_Attributes.attribute(idx).name(); 
	    double ORc = Math.exp(m_Par[j]);
	    result += " " + ((ORc > 1e10) ?  "" + ORc : Utils.doubleToString(ORc, 12, 4));
	}
	result += "\n";
	return result;
    }
    
    /**
     * Main method for testing this class.
     *
     * @param argv should contain the command line arguments to the
     * scheme (see Evaluation)
     */
    public static void main(String [] argv) {
	try {
	    System.out.println(MIEvaluation.evaluateModel(new MILRGEOM(), argv));
	} catch (Exception e) {
	    e.printStackTrace();
	  System.err.println(e.getMessage());
	}
    }
}


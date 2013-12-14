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
 *    DD.java
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
 * Re-implement the Diverse Density algorithm, changes the testing procedure.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.13 $ 
 */
public class DD extends MIClassifier implements OptionHandler {
    
    /** The index of the class attribute */
    protected int m_ClassIndex;
    
    protected double[] m_Par;

    /** The number of the class labels */
    protected int m_NumClasses;
    protected int m_IdIndex;    
    
    /** The filter used to make attributes numeric. */
    // private NominalToBinaryFilter m_NominalToBinary;
    
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
    
    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(1);
	newVector.addElement(new Option("\tTurn on debugging output.",
					"D", 0, "-D"));
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
      
    private class OptEng extends Optimization
    {
	/** 
	 * Evaluate objective function
	 * @param x the current values of variables
	 * @return the value of the objective function 
	 */
	protected double objectiveFunction(double[] x){
	    double nll = 0; // -LogLikelihood
	    for(int i=0; i<m_Classes.length; i++){ // ith bag
		int nI = m_Data[i][0].length; // numInstances in ith bag
		double bag = 0.0;  // NLL of pos bag
		
		for(int j=0; j<nI; j++){
		    double ins=0.0;
		    for(int k=0; k<m_Data[i].length; k++)
			ins += (m_Data[i][k][j]-x[k*2])*(m_Data[i][k][j]-x[k*2])*
			    x[k*2+1]*x[k*2+1];
		    ins = Math.exp(-ins);
		    ins = 1.0-ins;

		    if(m_Classes[i] == 1)
			bag += Math.log(ins);
		    else{
			if(ins<=m_Zero) ins=m_Zero;
			nll -= Math.log(ins);
		    }   
		}		
		
		if(m_Classes[i] == 1){
		    bag = 1.0 - Math.exp(bag);
		    if(bag<=m_Zero) bag=m_Zero;
		    nll -= Math.log(bag);
		}
	    }		
	    //System.out.println("??????NLL="+nll);
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
				
		double denom=0.0;	
		double[] numrt = new double[x.length];
		
		for(int j=0; j<nI; j++){
		    double exp=0.0;
		    for(int k=0; k<m_Data[i].length; k++)
			exp += (m_Data[i][k][j]-x[k*2])*(m_Data[i][k][j]-x[k*2])
			    *x[k*2+1]*x[k*2+1];			
		    exp = Math.exp(-exp);
		    exp = 1.0-exp;
		    if(m_Classes[i]==1)
			denom += Math.log(exp);		   		    

		    if(exp<=m_Zero) exp=m_Zero;
		    // Instance-wise update
		    for(int p=0; p<m_Data[i].length; p++){  // pth variable
			numrt[2*p] += (1.0-exp)*2.0*(x[2*p]-m_Data[i][p][j])*x[p*2+1]*x[p*2+1]
			    /exp;
			numrt[2*p+1] += 2.0*(1.0-exp)*(x[2*p]-m_Data[i][p][j])*(x[2*p]-m_Data[i][p][j])
			    *x[p*2+1]/exp;
		    }					    
		}		    
		
		// Bag-wise update 
		denom = 1.0-Math.exp(denom);
		if(denom <= m_Zero) denom = m_Zero;
		for(int q=0; q<m_Data[i].length; q++){
		    if(m_Classes[i]==1){
			grad[2*q] += numrt[2*q]*(1.0-denom)/denom;
			grad[2*q+1] += numrt[2*q+1]*(1.0-denom)/denom;
		    }else{
			grad[2*q] -= numrt[2*q];
			grad[2*q+1] -= numrt[2*q+1];
		    }
		}
	    } // one bag
	    	   
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
	FastVector maxSzIdx=new FastVector();
	int maxSz=0;
	
	m_Data  = new double [nC][nR][];              // Data values
	m_Classes  = new int [nC];                    // Class values
	m_Attributes = new Instances(train.exemplar(0).getInstances(),0);	
	if (m_Debug) {
	    System.out.println("Extracting data...");
	}
	
	for(int h=0; h<nC; h++){
	    Exemplar current = train.exemplar(h);
	    m_Classes[h] = (int)current.classValue();  // Class value starts from 0
	    Instances currInsts = current.getInstances();
	    int nI = currInsts.numInstances();
	    if(m_Classes[h]==1){
		if(nI>maxSz){
		    maxSz=nI;
		    maxSzIdx=new FastVector(1);
		    maxSzIdx.addElement(new Integer(h));
		}
		else if(nI == maxSz)
		    maxSzIdx.addElement(new Integer(h));
	    }
	    int idx=0;
	    for (int i = 0; i < train.numAttributes(); i++) {  		
		if ((i==m_ClassIndex) || (i==m_IdIndex))
		    continue;
		
		// initialize m_data[][][]		
		m_Data[h][idx] = new double[nI];
		for (int k=0; k<nI; k++)
		    m_Data[h][idx][k] = currInsts.instance(k).value(i);
		
		idx++;
	    }    	    
	}	
	
	if (m_Debug) {
	    System.out.println("\nIteration History..." );
	}
	
	double[] x = new double[nR*2], tmp = new double[x.length];
	double[][] b = new double[2][x.length]; 
	    
	OptEng opt;
	double nll, bestnll = Double.MAX_VALUE;
	for (int t=0; t<x.length; t++){
	    b[0][t] = Double.NaN; 
	    b[1][t] = Double.NaN;
	}
	
	// Largest Positive exemplar
	for(int s=0; s<maxSzIdx.size(); s++){
	    int exIdx = ((Integer)maxSzIdx.elementAt(s)).intValue();
	    for(int p=0; p<m_Data[exIdx][0].length; p++){
		for (int q=0; q < nR;q++){
		    x[2*q] = m_Data[exIdx][q][p];  // pick one instance
		    x[2*q+1] = 1;
		}
		
		opt = new OptEng();	
		//opt.setDebug(m_Debug);
		tmp = opt.findArgmin(x, b);
		while(tmp==null){
		    tmp = opt.getVarbValues();
		    if (m_Debug)
			System.out.println("200 iterations finished, not enough!");
		    tmp = opt.findArgmin(tmp, b);
		}
		nll = opt.getMinFunction();
		
		if(nll < bestnll){
		    bestnll = nll;
		    m_Par = tmp;
		    tmp = new double[x.length]; // Save memory
		    if (m_Debug)
			System.out.println("!!!!!!!!!!!!!!!!Smaller NLL found: "+nll);
		}
		if (m_Debug)
		    System.out.println(exIdx+":  -------------<Converged>--------------");
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
	
	// Extract the data
	Instances ins = exmp.getInstances();
	int nI = ins.numInstances(), nA = ins.numAttributes()-2;
	double[][] dat = new double [nI][nA];
	for(int j=0; j<nI; j++){
	    int idx=0;
	    for(int k=0; k<nA+2; k++){ 
		if((k==m_ClassIndex) || (k==m_IdIndex))
		    continue;
		dat[j][idx] = ins.instance(j).value(k);
		idx++;
	    }
	}
	
	// Compute the probability of the bag
	double [] distribution = new double[2];
	distribution[0]=0.0;  // log-Prob. for class 0
	
	for(int i=0; i<nI; i++){
	    double exp = 0.0;
	    for(int r=0; r<nA; r++)
		exp += (m_Par[r*2]-dat[i][r])*(m_Par[r*2]-dat[i][r])*
		    m_Par[r*2+1]*m_Par[r*2+1];
	    exp = Math.exp(-exp);
	    
	    // Prob. updated for one instance
	    distribution[0] += Math.log(1.0-exp);
	}

	distribution[0] = Math.exp(distribution[0]);
	distribution[1] = 1.0-distribution[0];	
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
	String result = "Diverse Density";
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
	    + "Variable       Point       Scale\n";
	for (int j = 0, idx=0; j < m_Par.length/2; j++, idx++) {
	    if((idx == m_ClassIndex) || (idx==m_IdIndex))
		idx++;
	    result += m_Attributes.attribute(idx).name();
	    result += " "+Utils.doubleToString(m_Par[j*2], 12, 4); 
	    result += " "+Utils.doubleToString(m_Par[j*2+1], 12, 4)+"\n";
	}
	
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
	    System.out.println(MIEvaluation.evaluateModel(new DD(), argv));
	} catch (Exception e) {
	    e.printStackTrace();
	    System.err.println(e.getMessage());
	}
    }
    
    public static double divBy100(double d){
	return d/100.0;
    }
}

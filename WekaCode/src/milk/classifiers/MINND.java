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
 *    MINND.java
 *    Copyright (C) 2001 Xin Xu
 *
 */

package milk.classifiers;
import milk.core.*;

import weka.classifiers.*;
import weka.core.*;
import java.lang.*;
import java.util.*;

/** 
 * 0657.591B Dissertation
 *
 * Multiple-Instance Nearest Neighbour with Distribution learner . <p>
 *
 * It uses gradient descent to find the weight for each dimension of each
 * exeamplar from the starting point of 1.0.  In order to
 * avoid overfitting, it uses mean-square function (i.e. the Euclidean 
 * distance) to search for the weights. <br>
 * It then uses the weights to cleanse the training data.  After that it 
 * searches for the weights again from the starting points of the weights
 * searched before. <br>
 * Finally it uses the most updated weights to cleanse the test exemplar
 * and then finds the nearest neighbour of the test exemplar using 
 * partly-weighted Kullback distance.  But the variances in the Kullback
 * distance are the ones before cleansing.
 * <p>Details see the 591 dissertation of Xin Xu.
 *
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.1 $
 */
public class MINND extends MIClassifier implements OptionHandler{

    /** The number of nearest neighbour for prediction */
    protected int m_Neighbour = 1;
 
    /** The mean for each attribute of each exemplar */
    protected double[][] m_Mean = null;

    /** The variance for each attribute of each exemplar */
    protected double[][] m_Variance = null;

    /** The dimension of each exemplar, i.e. (numAttributes-2) */
    protected int m_Dimension = 0;

    /** The class label of each exemplar */
    protected double[] m_Class = null;

    /** The number of class labels in the data */
    protected int m_NumClasses = 0;

    /** The weight of each exemplar */
    protected double[] m_Weights = null;

    /** The very small number representing zero */
    static private double m_ZERO = 1.0e-45;
    
    /** The learning rate in the gradient descent */
    protected double m_Rate = -1;
    
    /** The minimum values for numeric attributes. */
    private double [] m_MinArray=null;

    /** The maximum values for numeric attributes. */
    private double [] m_MaxArray=null;

    /** The stopping criteria of gradient descent*/
    private double m_STOP = 1.0e-45;
    
    /** The weights that alter the dimnesion of each exemplar */
    private double[][] m_Change=null;
    
    /** The noise data of each exemplar */
    private double[][] m_NoiseM= null, m_NoiseV=null, m_ValidM=null, m_ValidV=null;
    
    /** The number of nearest neighbour instances in the selection of noises 
	in the training data*/
    private int m_Select = 1;
    
    /** The number of nearest neighbour exemplars in the selection of noises 
	in the test data */
    private int m_Choose = 1;
    
    /** The class and ID attribute index of the data */
    private int m_ClassIndex, m_IdIndex;  
    
    /** The decay rate of learning rate */
    private double m_Decay = 0.5;
    
    /**
     * As normal Nearest Neighbour algorithm does, it's lazy and simply
     * records the exemplar information (i.e. mean and variance for each
     * dimension of each exemplar and their classes) when building the model.
     * There is actually no need to store the exemplars themselves.
     *
     * @param exs the training exemplars
     * @exception if the model cannot be built properly
     */    
    public void buildClassifier(Exemplars exs)throws Exception{
	Exemplars data = new Exemplars(exs);
	m_ClassIndex = data.classIndex();
	m_IdIndex = data.idIndex();
	int numegs = data.numExemplars();
	m_Dimension = data.numAttributes() - 2;
	m_Change = new double[numegs][m_Dimension];
	m_NumClasses = exs.numClasses();
	m_Mean = new double[numegs][m_Dimension];
	m_Variance = new double[numegs][m_Dimension];
	m_Class = new double[numegs];
	m_Weights = new double[numegs];
	m_NoiseM = new double[numegs][m_Dimension];
	m_NoiseV = new double[numegs][m_Dimension];
	m_ValidM = new double[numegs][m_Dimension];
	m_ValidV = new double[numegs][m_Dimension];
	m_MinArray = new double[m_Dimension];
	m_MaxArray = new double[m_Dimension];
	for(int v=0; v < m_Dimension; v++)
	    m_MinArray[v] = m_MaxArray[v] = Double.NaN;
	
	for(int w=0; w < numegs; w++){
	    updateMinMax(data.exemplar(w));
	}
	
	// Scale exemplars
	Exemplars newData = new Exemplars(data);
	data = new Exemplars(newData, numegs);
	for(int x=0; x < numegs; x++){
	    Exemplar example = newData.exemplar(x);
	    example = scale(example);
	    m_Mean[x] = example.meanOrMode();	
	    m_Variance[x] = example.variance();
	    for(int y=0; y < m_Variance[x].length; y++){
		if(Utils.eq(m_Variance[x][y],0.0))
		    m_Variance[x][y] = m_ZERO;
		m_Change[x][y] = 1.0;
	    }	
	    
	    data.add(example);
	    m_Class[x] = example.classValue();
	    m_Weights[x] = example.weight();	
	}
	
	for(int z=0; z < numegs; z++)
	    findWeights(z, m_Mean);
	
	// Pre-process and record "true estimated" parameters for distributions 
	for(int x=0; x < numegs; x++){
	    Exemplar example = preprocess(data, x);
	    System.out.println("???Exemplar "+x+" has been pre-processed:"+
			       data.exemplar(x).getInstances().sumOfWeights()+
			       "|"+example.getInstances().sumOfWeights()+
			       "; class:"+m_Class[x]);
	    if(Utils.gr(example.getInstances().sumOfWeights(), 0)){	
		m_ValidM[x] = example.meanOrMode();
		m_ValidV[x] = example.variance();
		for(int y=0; y < m_ValidV[x].length; y++){
		    if(Utils.eq(m_ValidV[x][y],0.0))
			m_ValidV[x][y] = m_ZERO;
		}	
	    }
	    else{
		m_ValidM[x] = null;
		m_ValidV[x] = null;
	    }
	}
	
	for(int z=0; z < numegs; z++)
	  if(m_ValidM[z] != null)
	    findWeights(z, m_ValidM);	
	
    }
    
    /**
     * Pre-process the given exemplar according to the other exemplars 
     * in the given exemplars.  It also updates noise data statistics.
     *
     * @param data the whole exemplars
     * @param pos the position of given exemplar in data
     * @return the processed exemplar
     * @exception if the returned exemplar is wrong 
     */
    public Exemplar preprocess(Exemplars data, int pos)
	throws Exception{
	Exemplar before = data.exemplar(pos);
	if((int)before.classValue() == 0){
	    m_NoiseM[pos] = null;
	    m_NoiseV[pos] = null;
	    return before;
	}
	
	Exemplar after = new Exemplar(before, 0);
	Exemplar noises =  new Exemplar(before, 0);
	
	for(int g=0; g < before.getInstances().numInstances(); g++){
	    Instance datum = before.getInstances().instance(g);
	    double[] dists = new double[data.numExemplars()];
	    
	    for(int i=0; i < data.numExemplars(); i++){
		if(i != pos)
		    dists[i] = distance(datum, m_Mean[i], m_Variance[i], i);
		else
		    dists[i] = Double.POSITIVE_INFINITY;
	    }		   

	    int[] pred = new int[m_NumClasses];
	    for(int n=0; n < pred.length; n++)
		pred[n] = 0;
	    
	    for(int o=0; o<m_Select; o++){
		int index = Utils.minIndex(dists);
		pred[(int)m_Class[index]]++;
		dists[index] = Double.POSITIVE_INFINITY;
	    }

	    int clas = Utils.maxIndex(pred);
	    if((int)datum.classValue() != clas)
		noises.add(datum);
	    else
		after.add(datum);		
	}
	
	if(Utils.gr(noises.getInstances().sumOfWeights(), 0)){	
	    m_NoiseM[pos] = noises.meanOrMode();
	    m_NoiseV[pos] = noises.variance();
	    for(int y=0; y < m_NoiseV[pos].length; y++){
		if(Utils.eq(m_NoiseV[pos][y],0.0))
		    m_NoiseV[pos][y] = m_ZERO;
	    }	
	}
	else{
	    m_NoiseM[pos] = null;
	    m_NoiseV[pos] = null;
	}
	
	return after;
    }
    
    /**
     * Calculates the distance between two instances
     *
     * @param first the first instance
     * @param second the second instance
     * @return the distance between the two given instances
     */          
    private double distance(Instance first, double[] mean, double[] var, int pos) {
	
	double diff, distance = 0;
	int j=0;
	for(int i = 0; i < first.numAttributes(); i++) { 
	    // Skipp nominal attributes (incl. class & ID)
	    if ((i == m_ClassIndex) || (i == m_IdIndex))
		continue;
	    
	    // If attribute is numeric
	    if(first.attribute(i).isNumeric()){
		if (!first.isMissing(i)){      
		    diff = first.value(i) - mean[j];
		    if(Utils.gr(var[j], m_ZERO))
			distance += m_Change[pos][j] * var[j] * diff * diff;
		    else
			distance += m_Change[pos][j] * diff * diff; 
		}
		else{
		    if(Utils.gr(var[j], m_ZERO))
			distance += m_Change[pos][j] * var[j];
		    else
			distance += m_Change[pos][j] * 1.0;
		}
	    }
	    j++;
	}
	
	return distance;
    }
    
    /**
     * Updates the minimum and maximum values for all the attributes
     * based on a new exemplar.
     *
     * @param ex the new exemplar
     */
    private void updateMinMax(Exemplar ex) {	
	Instances insts = ex.getInstances();
	int m=0;
	for (int j = 0;j < insts.numAttributes(); j++) {
	    if((j != ex.idIndex()) && (j != ex.classIndex())){
		if (insts.attribute(j).isNumeric()){
		    for(int k=0; k < insts.numInstances(); k++){
			Instance ins = insts.instance(k);
			if(!ins.isMissing(j)){
			    if (Double.isNaN(m_MinArray[m])) {
				m_MinArray[m] = ins.value(j);
				m_MaxArray[m] = ins.value(j);
			    } else {
				if (ins.value(j) < m_MinArray[m])
				    m_MinArray[m] = ins.value(j);
				else if (ins.value(j) > m_MaxArray[m])
				    m_MaxArray[m] = ins.value(j);
			    }
			}
		    } 
		}
		m++;
	    }
	}
    }
    
    /**
     * Scale the given exemplar so that the returned exemplar
     * has the value of 0 to 1 for each dimension
     * 
     * @param before the given exemplar
     * @return the resultant exemplar after scaling
     * @exception if given exampler cannot be scaled properly
     */
    private Exemplar scale(Exemplar before) throws Exception{
	Instances data = before.getInstances();
	Exemplar after = new Exemplar(before, 0);
	for(int i=0; i < data.numInstances(); i++){
	    Instance datum = data.instance(i);
	    Instance inst = (Instance)datum.copy();
	    int k=0;
	    for(int j=0; j < data.numAttributes(); j++){
		if((j != before.idIndex()) && (j != before.classIndex())){  
		    if(data.attribute(j).isNumeric())
			inst.setValue(j, (datum.value(j) - m_MinArray[k])/(m_MaxArray[k] - m_MinArray[k]));
		    k++;
		}
	    }
	    after.add(inst);
	}
	return after;
    }
    
    /**
     * Use gradient descent to distort the MU parameter for
     * the exemplar.  The exemplar can be in the specified row in the 
     * given matrix, which has numExemplar rows and numDimension columns;
     * or not in the matrix.
     * 
     * @param row the given row index
     * @return the result after gradient descent
     */
    public void findWeights(int row, double[][] mean){
	
	double[] neww = new double[m_Dimension];
	double[] oldw = new double[m_Dimension];
	System.arraycopy(m_Change[row], 0, neww, 0, m_Dimension);
	//for(int z=0; z<m_Dimension; z++)
	//System.out.println("mu("+row+"): "+origin[z]+" | "+newmu[z]);
	double newresult = target(neww, mean, row, m_Class);
	double result = Double.POSITIVE_INFINITY;
	double rate= 0.05;
	if(m_Rate != -1)
	    rate = m_Rate;
	System.out.println("???Start searching ...");
    search: 
	while(Utils.gr((result-newresult), m_STOP)){ // Full step
	    oldw = neww;
	    neww= new double[m_Dimension];
	    
	    double[] delta = delta(oldw, mean, row, m_Class);
	    
	    for(int i=0; i < m_Dimension; i++)
		if(Utils.gr(m_Variance[row][i], 0.0))
		    neww[i] = oldw[i] + rate * delta[i];
	    
	    result = newresult;
	    newresult = target(neww, mean, row, m_Class);
	    
	    //System.out.println("???old: "+result+"|new: "+newresult);
	    while(Utils.gr(newresult, result)){ // Search back
		//System.out.println("search back");
		if(m_Rate == -1){
		    rate *= m_Decay; // Decay
		    for(int i=0; i < m_Dimension; i++)
			if(Utils.gr(m_Variance[row][i], 0.0))
			    neww[i] = oldw[i] + rate * delta[i];
		    newresult = target(neww, mean, row, m_Class);
		}
		else{
		    for(int i=0; i < m_Dimension; i++)
			neww[i] = oldw[i];
		    break search;
		}
	    }
	}
	System.out.println("???Stop");
	m_Change[row] = neww;
    }
    
    /**
     * Delta of x in one step of gradient descent:
     * delta(Wij) = 1/2 * sum[k=1..N, k!=i](sqrt(P)*(Yi-Yk)/D - 1) * (MUij - MUkj)^2
     * where D = sqrt(sum[j=1..P]Kkj(MUij - MUkj)^2)
     * N is number of exemplars and P is number of dimensions
     *
     * @param x the weights of the exemplar in question
     * @param rowpos row index of x in X
     * @param Y the observed class label
     * @return the delta for all dimensions
     */
    private double[] delta(double[] x, double[][] X, int rowpos, double[] Y){
	double y = Y[rowpos];
	
	double[] delta=new double[m_Dimension];
	for(int h=0; h < m_Dimension; h++)
	    delta[h] = 0.0;
	
	for(int i=0; i < X.length; i++){
	    if((i != rowpos) && (X[i] != null)){
		double var = (y==Y[i]) ? 0.0 : Math.sqrt((double)m_Dimension - 1);
		double distance=0;
		for(int j=0; j < m_Dimension; j++)
		    if(Utils.gr(m_Variance[rowpos][j], 0.0))
			distance += x[j]*(X[rowpos][j]-X[i][j]) * (X[rowpos][j]-X[i][j]);
		distance = Math.sqrt(distance);
		if(distance != 0)
		    for(int k=0; k < m_Dimension; k++)
			if(m_Variance[rowpos][k] > 0.0)
			    delta[k] += (var/distance - 1.0) * 0.5 *
				(X[rowpos][k]-X[i][k]) *
				(X[rowpos][k]-X[i][k]);
	    }
	}
	//System.out.println("???delta: "+delta);
	return delta;
    }
    
    /**
     * Compute the target function to minimize in gradient descent
     * The formula is:<br>
     * 1/2*sum[i=1..p](f(X, Xi)-var(Y, Yi))^2 <p>
     * where p is the number of exemplars and Y is the class label.
     * In the case of X=MU, f() is the Euclidean distance between two
     * exemplars together with the related weights and var() is 
     * sqrt(numDimension)*(Y-Yi) where Y-Yi is either 0 (when Y==Yi)
     * or 1 (Y!=Yi) 
     *
     * @param x the weights of the exemplar in question
     * @param rowpos row index of x in X
     * @param Y the observed class label
     * @return the result of the target function
     */
    public double target(double[] x, double[][] X, int rowpos, double[] Y){
	double y = Y[rowpos], result=0;
	
	for(int i=0; i < X.length; i++){
	    if((i != rowpos) && (X[i] != null)){
		double var = (y==Y[i]) ? 0.0 : Math.sqrt((double)m_Dimension - 1);
		double f=0;
		for(int j=0; j < m_Dimension; j++)
		    if(Utils.gr(m_Variance[rowpos][j], 0.0)){
			f += x[j]*(X[rowpos][j]-X[i][j]) * (X[rowpos][j]-X[i][j]);     
			//System.out.println("i:"+i+" j: "+j+" row: "+rowpos);
		    }
		f = Math.sqrt(f);
		//System.out.println("???distance between "+rowpos+" and "+i+": "+f+"|y:"+y+" vs "+Y[i]);
		if(Double.isInfinite(f))
		    System.exit(1);
		result += 0.5 * (f - var) * (f - var);
	    }
	}
	//System.out.println("???target: "+result);
	return result;
    }    

    /**
     * Use Kullback Leibler distance to find the nearest neighbours of
     * the given exemplar.
     * It also uses K-Nearest Neighbour algorithm to classify the 
     * test exemplar
     *
     * @param ex the given test exemplar
     * @return the classification 
     * @exception Exception if the exemplar could not be classified
     * successfully
     */
    public double classifyExemplar(Exemplar e)throws Exception{
	Exemplar ex = new Exemplar(e);
	ex = scale(ex);
	
	double[] var = ex.variance();		
	// The Kullback distance to all exemplars
	double[] kullback = new double[m_Class.length];
	
	// The first K nearest neighbours' predictions */
	double[] predict = new double[m_NumClasses];
	for(int h=0; h < predict.length; h++)
	    predict[h] = 0;
	ex = cleanse(ex);
	
	if(ex.getInstances().numInstances() == 0){
	    System.out.println("???Whole exemplar falls into ambiguous area!");
	    return 1.0;                          // Bias towards positive class
	}
	
	double[] mean = ex.meanOrMode();	
	
	// Avoid zero sigma
	for(int h=0; h < var.length; h++){
	    if(Utils.eq(var[h],0.0))
		var[h] = m_ZERO;
	}	
	
	for(int i=0; i < m_Class.length; i++){
	    if(m_ValidM[i] != null)
		kullback[i] = kullback(mean, m_ValidM[i], var, m_Variance[i], i);
	    else
		kullback[i] = Double.POSITIVE_INFINITY;
	}
	
	for(int j=0; j < m_Neighbour; j++){
	    int pos = Utils.minIndex(kullback);
	    predict[(int)m_Class[pos]] += m_Weights[pos];	   
	    kullback[pos] = Double.POSITIVE_INFINITY;
	}	
	
	System.out.println("???There are still some unambiguous instances in this exemplar! Predicted as: "+Utils.maxIndex(predict));
	return (double)Utils.maxIndex(predict);	
    } 

    /**
     * Cleanse the given exemplar according to the valid and noise data
     * statistics
     *
     * @param before the given exemplar
     * @return the processed exemplar
     * @exception if the returned exemplar is wrong 
     */
    public Exemplar cleanse(Exemplar before) throws Exception{
	Exemplar after = new Exemplar(before, 0);
	
	for(int g=0; g < before.getInstances().numInstances(); g++){
	    Instance datum = before.getInstances().instance(g);
	    double[] minNoiDists = new double[m_Choose];
	    double[] minValDists = new double[m_Choose];
	    int noiseCount = 0, validCount = 0;
	    double[] nDist = new double[m_Mean.length]; 
	    double[] vDist = new double[m_Mean.length]; 
	    
	    for(int h=0; h < m_Mean.length; h++){
		if(m_ValidM[h] == null)
		    vDist[h] = Double.POSITIVE_INFINITY;
		else
		    vDist[h] = distance(datum, m_ValidM[h], m_ValidV[h], h);
		
		if(m_NoiseM[h] == null)
		    nDist[h] = Double.POSITIVE_INFINITY;
		else
		    nDist[h] = distance(datum, m_NoiseM[h], m_NoiseV[h], h);
	    }
	    
	    for(int k=0; k < m_Choose; k++){
		int pos = Utils.minIndex(vDist);
		minValDists[k] = vDist[pos];
		vDist[pos] = Double.POSITIVE_INFINITY;
		pos = Utils.minIndex(nDist);
		minNoiDists[k] = nDist[pos];
		nDist[pos] = Double.POSITIVE_INFINITY;
	    }
	    
	    int x = 0,y = 0;
	    while((x+y) < m_Choose){
		if(minValDists[x] <= minNoiDists[y]){
		    validCount++;
		    x++;
		}
		else{
		    noiseCount++;
		    y++;
		}
	    }
	    if(x >= y)
		after.add(datum);
	}
	
	return after;
    }    
    
    /**
     * This function calculates the Kullback Leibler distance between
     * two normal distributions.  This distance is always positive. 
     * Kullback Leibler distance = integral{f(X)ln(f(X)/g(X))}
     * Note that X is a vector.  Since we assume dimensions are independent
     * f(X)(g(X) the same) is actually the product of normal density
     * functions of each dimensions.  Also note that it should be log2
     * instead of (ln) in the formula, but we use (ln) simply for computational
     * convenience.
     *
     * The result is as follows, suppose there are P dimensions, and f(X)
     * is the first distribution and g(X) is the second:
     * Kullback = sum[1..P](ln(SIGMA2/SIGMA1)) +
     *            sum[1..P](SIGMA1^2 / (2*(SIGMA2^2))) +
     *            sum[1..P]((MU1-MU2)^2 / (2*(SIGMA2^2))) -
     *            P/2
     *
     * @param mu1 mu of the first normal distribution
     * @param mu2 mu of the second normal distribution 
     * @param var1 variance(SIGMA^2) of the first normal distribution
     * @param var2 variance(SIGMA^2) of the second normal distribution
     * @return the Kullback distance of two distributions
     */
    public double kullback(double[] mu1, double[] mu2,
			   double[] var1, double[] var2, int pos){
	int p = mu1.length;
	double result = 0;
	
	for(int y=0; y < p; y++){
	    if((Utils.gr(var1[y], 0)) && (Utils.gr(var2[y], 0))){
		result +=  
		    ((Math.log(Math.sqrt(var2[y]/var1[y]))) +
		     (var1[y] / (2.0*var2[y])) + 
		     (m_Change[pos][y] * (mu1[y]-mu2[y])*(mu1[y]-mu2[y]) / (2.0*var2[y])) -
		     0.5);
	    }
	}
	 
	return result;
    }
    
    /**
     * Returns an enumeration describing the available options
     * Valid options are: <p>
     *
     * -K number <br>
     * Set number of nearest neighbour used for prediction 
     * (Default: 1) <p>
     * 
     * -S number <br>
     * Set number of nearest neighbour instances used for cleansing the 
     * training data 
     * (Default: 1) <p>
     *
     * -E number <br>
     * Set number of nearest neighbour exemplars used for cleansing the 
     * testing data 
     * (Default: 1) <p>
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(3);
	
	newVector.addElement(new Option("\tSet number of nearest neighbour\n" +
					"\tfor prediction\n" +
					"\t(default 1)","K", 1, "-K <number of"
					+"neighbours>"));
	newVector.addElement(new Option("\tSet number of nearest neighbour\n" +
					"\tfor cleansing the training data\n" +
					"\t(default 1)","S", 1, "-S <number of"
					+"neighbours>"));
	newVector.addElement(new Option("\tSet number of nearest neighbour\n" +
					"\tfor cleansing the testing data\n" +
					"\t(default 1)","E", 1, "-E <number of"
					+"neighbours>"));
	return newVector.elements();
    }
    
    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception{
	
	String numNeighbourString = Utils.getOption('K', options);
	if (numNeighbourString.length() != 0) 
	    m_Neighbour = Integer.parseInt(numNeighbourString);
	else 
	    m_Neighbour = 1;

        numNeighbourString = Utils.getOption('S', options);
	if (numNeighbourString.length() != 0) 
	    m_Select = Integer.parseInt(numNeighbourString);
	else 
	    m_Select = 1;

	numNeighbourString = Utils.getOption('E', options);
	if (numNeighbourString.length() != 0) 
	    m_Choose = Integer.parseInt(numNeighbourString);
	else 
	    m_Choose = 1;
    }
    
    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
	
	String [] options = new String [6];
	int current = 0;
	options[current++] = "-K"; options[current++] = "" + m_Neighbour;
	options[current++] = "-S"; options[current++] = "" + m_Select;
	options[current++] = "-E"; options[current++] = "" + m_Choose;
	while (current < options.length) 
	    options[current++] = "";
	return options;
    }

    /**
     * Main method for testing.
     *
     * @param args the options for the classifier
     */
    public static void main(String[] args) {	
	try {
	    System.out.println(MIEvaluation.evaluateModel(new MINND(), args));
	} catch (Exception e) {
	    e.printStackTrace();
	    System.err.println(e.getMessage());
	}
    }
}

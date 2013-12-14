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
 *    MIBoost.java  copyright(c) 2003 Eibe Frank, Xin Xu
 *
 */

package milk.classifiers;
import milk.core.*;
import weka.classifiers.trees.*;
import weka.classifiers.*;
import java.util.*;
import java.io.*;
import weka.core.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Discretize;
/**
 * 
 * MI AdaBoost method, consider the geometric mean of posterior
 * of instances inside a bag (arithmatic mean of log-posterior) and 
 * the expectation for a bag is taken inside the loss function.  
 * Exact derivation from Hastie et al. paper
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $ 
 */
public class MIBoost extends MIClassifier 
    implements OptionHandler {
    
    /** The index of the class attribute */
    protected int m_ClassIndex;
    
    protected Classifier[] m_Models;

    /** The number of the class labels */
    protected int m_NumClasses;
    protected int m_IdIndex;
     
    /** Debugging output */
    protected boolean m_Debug=false;
   
    /** Class labels for each bag */
    protected int[] m_Classes;
    
    /** All attribute names */
    protected Instances m_Attributes;
    
    /** Number of iterations */   
    private int m_NumIterations = 100;
    
    /** The model base classifier to use */
    protected Classifier m_Classifier = new weka.classifiers.trees.DecisionStump();    
    
    /** Voting weights of models */ 
    protected double[] m_Beta;
    
    protected int m_MaxIterations = 10;

    protected int m_DiscretizeBin = 0;
    protected Discretize m_Filter = null;
    
    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(6);
	newVector.addElement(new Option("\tTurn on debugging output.",
					"D", 0, "-D"));
	
	newVector.addElement(new Option("\tThe number of bins in discretization\n"
					+"\t(default 0, no discretization)",
					"B", 1, "-B <num>"));	
	
	newVector.addElement(new Option("\tMaximum number of boost iterations.\n"
					+"\t(default 10)",
					"R", 1, "-R <num>"));	
	
	newVector.addElement(new Option("\tFull name of classifier to boost.\n"
					+"\teg: weka.classifiers.bayes.NaiveBayes",
					"W", 1, "-W <class name>"));
	if ((m_Classifier != null) &&
	    (m_Classifier instanceof OptionHandler)) {
	    newVector.addElement(new Option("","", 0, 
					    "\nOptions specific to classifier "
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
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
	setDebug(Utils.getFlag('D', options));

	String bin = Utils.getOption('B', options);
	if (bin.length() != 0) {
	    setDiscretizeBin(Integer.parseInt(bin));
	} else {
	    setDiscretizeBin(0);
	}
	
	String boostIterations = Utils.getOption('R', options);
	if (boostIterations.length() != 0) {
	    setMaxIterations(Integer.parseInt(boostIterations));
	} else {
	    setMaxIterations(10);
	}
	
	String classifierName = Utils.getOption('W', options);
	if (classifierName.length() != 0) 
	    m_Classifier=Classifier.forName(classifierName,
					    Utils.partitionOptions(options));
	else{
	    //throw new Exception("A classifier must be specified with"
	    //			+ " the -W option.");
	    //m_Classifier = new weka.classifiers.trees.DecisionStump();    
	}	
    }
    
    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
	
	String [] classifierOptions = new String [0];
	if ((m_Classifier != null) && 
	    (m_Classifier instanceof OptionHandler)) {
	    classifierOptions = ((OptionHandler)m_Classifier).getOptions();
	}
	
	String [] options = new String [classifierOptions.length + 11];
	int current = 0;
	if (getDebug()) {
	    options[current++] = "-D";
	}
  
	options[current++] = "-R"; options[current++] = "" + getMaxIterations();
	options[current++] = "-B"; options[current++] = "" + getDiscretizeBin();
	
	if (m_Classifier != null) {
	    options[current++] = "-W";
	    options[current++] = m_Classifier.getClass().getName();
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
     * Set the classifier for boosting. 
     *
     * @param newClassifier the Classifier to use.
     */
    public void setClassifier(Classifier newClassifier) {	
	m_Classifier = newClassifier;
    }
    
    /**
     * Get the classifier used as the classifier
     *
     * @return the classifier used as the base classifier
     */
    public Classifier getClassifier() {	
	return m_Classifier;
    }  
    
    /**
     * Set the maximum number of boost iterations
     *
     * @param maxIterations the maximum number of boost iterations
     */
    public void setMaxIterations(int maxIterations) {	
	m_MaxIterations = maxIterations;
    }
    
    /**
     * Get the maximum number of boost iterations
     *
     * @return the maximum number of boost iterations
     */
    public int getMaxIterations() {
	
	return m_MaxIterations;
    }
    
    /**
     * Set the number of bins in discretization
     *
     * @param bin the number of bins in discretization
     */
    public void setDiscretizeBin(int bin) {	
	m_DiscretizeBin = bin;
    }
    
    /**
     * Get the number of bins in discretization
     *
     * @return the number of bins in discretization
     */
    public int getDiscretizeBin() {	
	return m_DiscretizeBin;
    }
    
    private class OptEng extends Optimization{
	private double[] weights, errs;
	
	public void setWeights(double[] w){
	    weights = w;
	}
	
	public void setErrs(double[] e){
	    errs = e;
	}
	
	/** 
	 * Evaluate objective function
	 * @param x the current values of variables
	 * @return the value of the objective function 
	 */
	protected double objectiveFunction(double[] x) throws Exception{
	    double obj=0;
	    for(int i=0; i<weights.length; i++){
		obj += weights[i]*Math.exp(x[0]*(2.0*errs[i]-1.0));
		if(Double.isNaN(obj))
		    throw new Exception("Objective function value is NaN!");
	
	    }
	    return obj;
	}
	
	/** 
	 * Evaluate Jacobian vector
	 * @param x the current values of variables
	 * @return the gradient vector 
	 */
	protected double[] evaluateGradient(double[] x)  throws Exception{
	    double[] grad = new double[1];
	    for(int i=0; i<weights.length; i++){
		grad[0] += weights[i]*(2.0*errs[i]-1.0)*Math.exp(x[0]*(2.0*errs[i]-1.0));
		if(Double.isNaN(grad[0]))
		    throw new Exception("Gradient is NaN!");
	
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
    public void buildClassifier(Exemplars exps) throws Exception {
	
	Exemplars train = new Exemplars(exps);

	if (train.classAttribute().type() != Attribute.NOMINAL) {
	    throw new Exception("Class attribute must be nominal.");
	}
	if (train.checkForStringAttributes()) {
	    throw new Exception("Can't handle string attributes!");
	}
	
	m_ClassIndex = train.classIndex();
	m_IdIndex = train.idIndex();
	m_NumClasses = train.numClasses();
	m_NumIterations = m_MaxIterations;

	if (m_NumClasses > 2) {
	    throw new Exception("Not yet prepared to deal with multiple classes!");
	}
	
	if (m_Classifier == null)
	    throw new Exception("A base classifier has not been specified!");
	if(!(m_Classifier instanceof WeightedInstancesHandler))
	     throw new Exception("Base classifier cannot handle weighted instances!");
	
	m_Models = Classifier.makeCopies(m_Classifier, getMaxIterations());
	if(m_Debug)
	    System.err.println("Base classifier: "+m_Classifier.getClass().getName());
		
	m_Beta = new double[m_NumIterations];
	m_Attributes = new Instances(train.exemplar(0).getInstances(),0);
	
	double N = (double)train.numExemplars(), sumNi=0;
	Instances data=new Instances(m_Attributes, 0);// Data to learn a model	
	data.deleteAttributeAt(m_IdIndex);// ID attribute useless	
	Instances dataset = new Instances(data,0);
	
	// Initialize weights
	for(int i=0; i<N; i++)
	    sumNi += train.exemplar(i).getInstances().numInstances();
	
	for(int i=0; i<N; i++){
	    Exemplar exi = train.exemplar(i);
	    exi.setWeight(sumNi/N);
	    Instances insts = exi.getInstances();
	    double ni = (double)insts.numInstances();    
	    for(int j=0; j<ni; j++){	
		Instance ins = new Instance(insts.instance(j));// Copy
		//insts.instance(j).setWeight(1.0);	
		
		ins.deleteAttributeAt(m_IdIndex);
		ins.setDataset(dataset);
		ins.setWeight(exi.weight()/ni);
		data.add(ins);
	    }
	}
	
	// Assume the order of the instances are preserved in the Discretize filter
	if(m_DiscretizeBin > 0){
	    m_Filter = new Discretize();
	    m_Filter.setInputFormat(new Instances(data, 0));
	    m_Filter.setBins(m_DiscretizeBin);
	    data = Filter.useFilter(data, m_Filter);
	}
	
	// Main algorithm
	int dataIdx;
    iterations:
	for(int m=0; m < m_MaxIterations; m++){
	    if(m_Debug)
		System.err.println("\nIteration "+m);
	    // Build a model
	    m_Models[m].buildClassifier(data);
	    
	    // Prediction of each bag
	    double[] err=new double[(int)N], weights=new double[(int)N];
	    boolean perfect = true, tooWrong=true;
	    dataIdx = 0;
	    for(int n=0; n<N; n++){
		Exemplar exn = train.exemplar(n);
		// Prediction of each instance and the predicted class distribution
		// of the bag		
		double nn = (double)exn.getInstances().numInstances();
		for(int p=0; p<nn; p++){
		    Instance testIns = data.instance(dataIdx++);			
		    if((int)m_Models[m].classifyInstance(testIns) 
		       != (int)exn.classValue()) // Weighted instance-wise 0-1 errors
			err[n] ++;		       		       
		}
		weights[n] = exn.weight();
		err[n] /= nn;
		if(err[n] > 0.5)
		    perfect = false;
		if(err[n] < 0.5)
		    tooWrong = false;
	    }
	    
	    if(perfect || tooWrong){ // No or 100% classification error, cannot find beta
		if (m == 0)
		    m_Beta[m] = 1.0;
		else		    
		    m_Beta[m] = 0;		
		m_NumIterations = m+1;
		if(m_Debug)  System.err.println("No errors");
		break iterations;
	    }
	    
	    double[] x = new double[1];
	    x[0] = 0;
	    double[][] b = new double[2][x.length];
	    b[0][0] = Double.NaN;
	    b[1][0] = Double.NaN;
	    
	    OptEng opt = new OptEng();	
	    opt.setWeights(weights);
	    opt.setErrs(err);
	    //opt.setDebug(m_Debug);
	    if (m_Debug)
		System.out.println("Start searching for c... ");
	    x = opt.findArgmin(x, b);
	    while(x==null){
		x = opt.getVarbValues();
		if (m_Debug)
		    System.out.println("200 iterations finished, not enough!");
		x = opt.findArgmin(x, b);
	    }	
	    if (m_Debug)
		System.out.println("Finished.");    
	    m_Beta[m] = x[0];
	    
	    if(m_Debug)
		System.err.println("c = "+m_Beta[m]);
	    
	    // Stop if error too small or error too big and ignore this model
	    if (Double.isInfinite(m_Beta[m]) 
		|| Utils.smOrEq(m_Beta[m], 0)
		) {
		if (m == 0)
		    m_Beta[m] = 1.0;
		else		    
		    m_Beta[m] = 0;
		m_NumIterations = m+1;
		if(m_Debug)
		    System.err.println("Errors out of range!");
		break iterations;
	    }
	    
	    // Update weights of data and class label of wfData
	    dataIdx=0;
	    double totWeights=0;
	    for(int r=0; r<N; r++){		
		Exemplar exr = train.exemplar(r);
		exr.setWeight(weights[r]*Math.exp(m_Beta[m]*(2.0*err[r]-1.0)));
		totWeights += exr.weight();
	    }
	    
	    if(m_Debug)
		System.err.println("Total weights = "+totWeights);

	    for(int r=0; r<N; r++){		
		Exemplar exr = train.exemplar(r);
		double num = (double)exr.getInstances().numInstances();
		exr.setWeight(sumNi*exr.weight()/totWeights);
		//if(m_Debug)
		//    System.err.print("\nExemplar "+r+"="+exr.weight()+": \t");
		for(int s=0; s<num; s++){
		    Instance inss = data.instance(dataIdx);	
		    inss.setWeight(exr.weight()/num);		   
		    //    if(m_Debug)
		    //  System.err.print("instance "+s+"="+inss.weight()+
		    //			 "|ew*iw*sumNi="+data.instance(dataIdx).weight()+"\t");
		    if(Double.isNaN(inss.weight()))
			throw new Exception("instance "+s+" in bag "+r+" has weight NaN!"); 
		    dataIdx++;
		}
		//if(m_Debug)
		//    System.err.println();
	    }	       
	}
    }		
    
    /**
     * Computes the distribution for a given exemplar
     *
     * @param exmp the exemplar for which distribution is computed
     * @return the classification
     * @exception Exception if the distribution can't be computed successfully
     */
    public double[] distributionForExemplar(Exemplar exmp) 
	throws Exception {
	double[] rt = new double[m_NumClasses];
	Instances insts = new Instances(exmp.getInstances());		
	double n = (double)insts.numInstances();
	insts.deleteAttributeAt(m_IdIndex);// ID attribute useless
	if(m_DiscretizeBin > 0)
	    insts = Filter.useFilter(insts, m_Filter);
	
	for(int y=0; y<n; y++){
	    Instance ins = insts.instance(y);	
	    for(int x=0; x<m_NumIterations; x++)
		rt[(int)m_Models[x].classifyInstance(ins)] += m_Beta[x]/n;
	}
	
	for(int i=0; i<rt.length; i++)
	    rt[i] = Math.exp(rt[i]);
	Utils.normalize(rt);
	return rt;
    }
    
    /**
     * Gets a string describing the classifier.
     *
     * @return a string describing the classifer built.
     */
    public String toString() {

      if (m_Models == null) {
	return "No model built yet!";
      }
	StringBuffer text = new StringBuffer();
	text.append("MIBoost: number of bins in discretization = "+m_DiscretizeBin+"\n");
	if (m_NumIterations == 0) {
	    text.append("No model built yet.\n");
	} else if (m_NumIterations == 1) {
	    text.append("No boosting possible, one classifier used: Weight = " 
			+ Utils.roundDouble(m_Beta[0], 2)+"\n");
	    text.append("Base classifiers:\n"+m_Models[0].toString());
	} else {
	    text.append("Base classifiers and their weights: \n");
	    for (int i = 0; i < m_NumIterations ; i++) {
		text.append("\n\n"+i+": Weight = " + Utils.roundDouble(m_Beta[i], 2)
			    +"\nBase classifier:\n"+m_Models[i].toString() );
	    }
	}
	
	text.append("\n\nNumber of performed Iterations: " 
		    + m_NumIterations + "\n");
	
	return text.toString();
    }
    
    /**
     * Main method for testing this class.
     *
     * @param argv should contain the command line arguments to the
     * scheme (see Evaluation)
     */
    public static void main(String [] argv) {
	try {
	    System.out.println(MIEvaluation.evaluateModel(new MIBoost(), argv));
	} catch (Exception e) {
	    e.printStackTrace();
	    System.err.println(e.getMessage());
	}
    }
}

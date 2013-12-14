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
 *    TLD.java
 *    Copyright (C) 2002 Eibe Frank, Xin Xu
 *
 */

package milk.classifiers;
import milk.core.*;

import weka.classifiers.*;
import weka.core.*;
import java.lang.*;
import java.util.*;

/** 
 * 0657.594 Thesis
 *
 * A simpler version of TLD, \mu random but \sigma^2 fixed and estimated via data <p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.1 $
 */
public class TLDSimple extends MIClassifier implements OptionHandler{

    /** The mean for each attribute of each positive exemplar */
    protected double[][] m_MeanP = null;

    /** The mean for each attribute of each negative exemplar */
    protected double[][] m_MeanN = null;

    /** The effective sum of weights of each positive exemplar in each dimension*/
    protected double[][] m_SumP = null;
    
    /** The effective sum of weights of each negative exemplar in each dimension*/
    protected double[][] m_SumN = null;

    /** Estimated sigma^2 in positive bags*/
    protected double[] m_SgmSqP;
    
    /** Estimated sigma^2 in negative bags*/
    protected double[] m_SgmSqN;
    
    /** The parameters to be estimated for each positive exemplar*/
    protected double[] m_ParamsP = null;
    
    /** The parameters to be estimated for each negative exemplar*/
    protected double[] m_ParamsN = null;
    
    /** The dimension of each exemplar, i.e. (numAttributes-2) */
    protected int m_Dimension = 0;
    
    /** The class label of each exemplar */
    protected double[] m_Class = null;
    
    /** The number of class labels in the data */
    protected int m_NumClasses = 2;
    
    /** The class and ID attribute index of the data */
    private int m_ClassIndex, m_IdIndex;  

    /** The very small number representing zero */
    static public double ZERO = 1.0e-12;   

    protected int m_Run = 1;

    protected long m_Seed = 1;

    protected double m_Cutoff;

    protected boolean m_UseEmpiricalCutOff = false;    

    private double[] m_LkRatio;

    private Instances m_Attribute = null;
    
    /**
     *
     * @param exs the training exemplars
     * @exception if the model cannot be built properly
     */    
    public void buildClassifier(Exemplars exs)throws Exception{
	m_ClassIndex = exs.classIndex();
	m_IdIndex = exs.idIndex();
	int numegs = exs.numExemplars();
	m_Dimension = exs.numAttributes() - 2;
	m_Attribute = new Instances(exs.exemplar(0).getInstances(),0);
	Exemplars pos = new Exemplars(exs, 0), neg = new Exemplars(exs, 0);
	
	// Divide into two groups
	for(int u=0; u<numegs; u++){
	    Exemplar example = exs.exemplar(u);
	    if(example.classValue() == 0)
		pos.add(example);
	    else
		neg.add(example);
	}	
	int pnum = pos.numExemplars(), nnum = neg.numExemplars();	
	
	// xBar, n
	m_MeanP = new double[pnum][m_Dimension];
	m_SumP = new double[pnum][m_Dimension];
	m_MeanN = new double[nnum][m_Dimension];
	m_SumN = new double[nnum][m_Dimension];
	// w, m
	m_ParamsP = new double[2*m_Dimension];
	m_ParamsN = new double[2*m_Dimension];
	// \sigma^2
	m_SgmSqP = new double[m_Dimension];
	m_SgmSqN = new double[m_Dimension];
	// S^2
	double[][] varP=new double[pnum][m_Dimension], 
	    varN=new double[nnum][m_Dimension];
	// numOfEx 'e' without all missing
	double[] effNumExP=new double[m_Dimension], 
	    effNumExN=new double[m_Dimension];
	// For the starting values
	double[] pMM=new double[m_Dimension], 
	    nMM=new double[m_Dimension],
	    pVM=new double[m_Dimension],
	    nVM=new double[m_Dimension];
	// # of exemplars with only one instance
	double[] numOneInsExsP=new double[m_Dimension],
	    numOneInsExsN=new double[m_Dimension];
	// sum_i(1/n_i)
	double[] pInvN = new double[m_Dimension], nInvN = new double[m_Dimension];
	
	// Extract metadata from both positive and negative bags
	for(int v=0; v < pnum; v++){
	    Exemplar px = pos.exemplar(v);
	    m_MeanP[v] = px.meanOrMode();
	    varP[v] = px.variance();
	    Instances pxi =  px.getInstances();
	    
	    for (int w=0,t=0; w < m_Dimension; w++,t++){		
		if((t==m_ClassIndex) || (t==m_IdIndex))
		    t++;	
		if(varP[v][w] <= 0.0)
		    varP[v][w] = 0.0;
		if(!Double.isNaN(m_MeanP[v][w])){

		    for(int u=0;u<pxi.numInstances();u++)
			if(!pxi.instance(u).isMissing(t))			    
			    m_SumP[v][w] += pxi.instance(u).weight();
		    
		    pMM[w] += m_MeanP[v][w];
		    pVM[w] += m_MeanP[v][w]*m_MeanP[v][w];		    
		    if((m_SumP[v][w]>1) && (varP[v][w]>ZERO)){			
			m_SgmSqP[w] += varP[v][w]*(m_SumP[v][w]-1.0)/m_SumP[v][w];
			//m_SgmSqP[w] += varP[v][w]*(m_SumP[v][w]-1.0);
			effNumExP[w]++; // Not count exemplars with 1 instance
			pInvN[w] += 1.0/m_SumP[v][w];
			//pInvN[w] += m_SumP[v][w];
		    }
		    else
			numOneInsExsP[w]++;
		}				
	    }	    		    
	}
	
	for(int v=0; v < nnum; v++){
	    Exemplar nx = neg.exemplar(v);
	    m_MeanN[v] = nx.meanOrMode();
	    varN[v] = nx.variance();
	    Instances nxi =  nx.getInstances();
	    
	    for (int w=0,t=0; w < m_Dimension; w++,t++){
		
		if((t==m_ClassIndex) || (t==m_IdIndex))
		    t++;	
		if(varN[v][w] <= 0.0)
		    varN[v][w] = 0.0;
		if(!Double.isNaN(m_MeanN[v][w])){
		    for(int u=0;u<nxi.numInstances();u++)
			if(!nxi.instance(u).isMissing(t))
			    m_SumN[v][w] += nxi.instance(u).weight();	
		    
		    nMM[w] += m_MeanN[v][w]; 
		    nVM[w] += m_MeanN[v][w]*m_MeanN[v][w];
		    if((m_SumN[v][w]>1) && (varN[v][w]>ZERO)){			
			m_SgmSqN[w] += varN[v][w]*(m_SumN[v][w]-1.0)/m_SumN[v][w];
			//m_SgmSqN[w] += varN[v][w]*(m_SumN[v][w]-1.0);
			effNumExN[w]++; // Not count exemplars with 1 instance
			nInvN[w] += 1.0/m_SumN[v][w];
			//nInvN[w] += m_SumN[v][w];
		    }
		    else
			numOneInsExsN[w]++;
		}					
	    }
	}
	
	// Expected \sigma^2
	for (int u=0; u < m_Dimension; u++){
	    // For exemplars with only one instance, use avg(\sigma^2) of other exemplars
	    m_SgmSqP[u] /= (effNumExP[u]-pInvN[u]);
	    m_SgmSqN[u] /= (effNumExN[u]-nInvN[u]);
	    //m_SgmSqP[u] /= (pInvN[u]-effNumExP[u]);
	    //m_SgmSqN[u] /= (nInvN[u]-effNumExN[u]);
	    effNumExP[u] += numOneInsExsP[u];
	    effNumExN[u] += numOneInsExsN[u];
	    pMM[u] /= effNumExP[u];
	    nMM[u] /= effNumExN[u];
	    pVM[u] = pVM[u]/(effNumExP[u]-1.0) - pMM[u]*pMM[u]*effNumExP[u]/(effNumExP[u]-1.0);
	    nVM[u] = nVM[u]/(effNumExN[u]-1.0) - nMM[u]*nMM[u]*effNumExN[u]/(effNumExN[u]-1.0);
	}
	
	//Bounds and parameter values for each run
	double[][] bounds = new double[2][2];
	double[] pThisParam = new double[2], 
	    nThisParam = new double[2];
	
	// Initial values for parameters
	double w, m;
	Random whichEx = new Random(m_Seed);
	
	// Optimize for one dimension
	for (int x=0; x < m_Dimension; x++){     
	    // System.out.println("\n\n!!!!!!!!!!!!!!!!!!!!!!???Dimension #"+x);
	    
	    // Positive examplars: first run 
	    pThisParam[0] = pVM[x];  // w
	    if( pThisParam[0] <= ZERO)
		pThisParam[0] = 1.0;
	    pThisParam[1] = pMM[x];  // m
	    
	    // Negative examplars: first run
	    nThisParam[0] = nVM[x];  // w
	    if(nThisParam[0] <= ZERO)
		nThisParam[0] = 1.0;
	    nThisParam[1] = nMM[x];  // m
	    
	    // Bound constraints
	    bounds[0][0] = ZERO; // w > 0
	    bounds[0][1] = Double.NaN;
	    bounds[1][0] = Double.NaN; 
	    bounds[1][1] = Double.NaN;

	    double pminVal=Double.MAX_VALUE, nminVal=Double.MAX_VALUE; 
	    TLDSimple_Optm pOp=null, nOp=null;	
	    boolean isRunValid = true;
	    double[] sumP=new double[pnum], meanP=new double[pnum];
	    double[] sumN=new double[nnum], meanN=new double[nnum];
	    
	    // One dimension
	    for(int p=0; p<pnum; p++){
		sumP[p] = m_SumP[p][x];
		meanP[p] = m_MeanP[p][x];
	    }
	    for(int q=0; q<nnum; q++){
		sumN[q] = m_SumN[q][x];
		meanN[q] = m_MeanN[q][x];
	    }
	    
	    for(int y=0; y<m_Run; y++){
		//System.out.println("\n\n!!!!!!!!!Positive exemplars: Run #"+y);
		double thisMin;
		pOp = new TLDSimple_Optm();
		pOp.setNum(sumP);
		pOp.setSgmSq(m_SgmSqP[x]);
		pOp.setXBar(meanP);
		//pOp.setDebug(true);
		pThisParam = pOp.findArgmin(pThisParam, bounds);
		while(pThisParam==null){
		    pThisParam = pOp.getVarbValues();		    
		    System.out.println("!!! 200 iterations finished, not enough!");
		    pThisParam = pOp.findArgmin(pThisParam, bounds);
		}	
		
		thisMin = pOp.getMinFunction();
		if(!Double.isNaN(thisMin) && (thisMin<pminVal)){
		    pminVal = thisMin;
		    for(int z=0; z<2; z++)
			m_ParamsP[2*x+z] = pThisParam[z];
		}
		
		if(Double.isNaN(thisMin)){
		    pThisParam = new double[2];
		    isRunValid =false;
		}
		if(!isRunValid){ y--; isRunValid=true; } 
		
		// Change the initial parameters and restart
		int pone = whichEx.nextInt(pnum);
		
		// Positive exemplars: next run 
		while(Double.isNaN(m_MeanP[pone][x]))
		    pone = whichEx.nextInt(pnum);
		
		m = m_MeanP[pone][x];
		w = (m-pThisParam[1])*(m-pThisParam[1]);
		pThisParam[0] = w;  // w
		pThisParam[1] = m;  // m	    
	    }
	    
	    for(int y=0; y<m_Run; y++){
		//System.out.println("\n\n!!!!!!!!!Negative exemplars: Run #"+y);
		double thisMin;
		nOp = new TLDSimple_Optm();
		nOp.setNum(sumN);
		nOp.setSgmSq(m_SgmSqN[x]);
		nOp.setXBar(meanN);
		//nOp.setDebug(true);
		nThisParam = nOp.findArgmin(nThisParam, bounds);
		while(nThisParam==null){
		    nThisParam = nOp.getVarbValues();
		    System.out.println("!!! 200 iterations finished, not enough!");
		    nThisParam = nOp.findArgmin(nThisParam, bounds);
		}			
		
		thisMin = nOp.getMinFunction();
		if(!Double.isNaN(thisMin) && (thisMin<nminVal)){
		    nminVal = thisMin;
		    for(int z=0; z<2; z++)
			m_ParamsN[2*x+z] = nThisParam[z];     
		}
		
		if(Double.isNaN(thisMin)){
		    nThisParam = new double[2];
		    isRunValid =false;
		}
		
		if(!isRunValid){ y--; isRunValid=true; } 		
		
		// Change the initial parameters and restart	   	    
		int none = whichEx.nextInt(nnum);// Randomly pick one pos. exmpl.
		
		// Negative exemplars: next run 
		while(Double.isNaN(m_MeanN[none][x]))
		    none = whichEx.nextInt(nnum);
		
		m = m_MeanN[none][x];
		w = (m-nThisParam[1])*(m-nThisParam[1]);
		nThisParam[0] = w;  // w
		nThisParam[1] = m;  // m	 		
	    }	    	    	    
	}
			
	m_LkRatio = new double[m_Dimension];
	
	if(m_UseEmpiricalCutOff){	
	    // Find the empirical cut-off
	    double[] pLogOdds=new double[pnum], nLogOdds=new double[nnum];  
	    for(int p=0; p<pnum; p++)
		pLogOdds[p] = 
		    likelihoodRatio(m_SumP[p], m_MeanP[p]);
	   
	    for(int q=0; q<nnum; q++)
		nLogOdds[q] = 
		    likelihoodRatio(m_SumN[q], m_MeanN[q]);
	    
	    // Update m_Cutoff
	    findCutOff(pLogOdds, nLogOdds);
	}
	else
	    m_Cutoff = -Math.log((double)pnum/(double)nnum);
	
	/* 
	for(int x=0, y=0; x<m_Dimension; x++, y++){
	    if((x==exs.classIndex()) || (x==exs.idIndex()))
		y++;
	    
	    w=m_ParamsP[2*x]; m=m_ParamsP[2*x+1];
	    System.err.println("\n\n???Positive: ( "+exs.attribute(y)+
			       "):  w="+w+", m="+m+", sgmSq="+m_SgmSqP[x]);
	    
	    w=m_ParamsN[2*x]; m=m_ParamsN[2*x+1];
	    System.err.println("???Negative: ("+exs.attribute(y)+
			       "):  w="+w+", m="+m+", sgmSq="+m_SgmSqN[x]+
			       "\nAvg. log-likelihood ratio in training data="
			       +(m_LkRatio[x]/(pnum+nnum)));
	}	
	*/
	System.err.println("\n\n???Cut-off="+m_Cutoff);
    }        
    
    /**
     *
     * @param ex the given test exemplar
     * @return the classification 
     * @exception Exception if the exemplar could not be classified
     * successfully
     */
    public double classifyExemplar(Exemplar e)throws Exception{
	Exemplar ex = new Exemplar(e);
	Instances exi = ex.getInstances();
	double[] n = new double[m_Dimension], xBar = ex.meanOrMode();
	
	for (int w=0, t=0; w < m_Dimension; w++, t++){
	    if((t==m_ClassIndex) || (t==m_IdIndex))
		t++;	
	    for(int u=0;u<exi.numInstances();u++)
		if(!exi.instance(u).isMissing(t))
		    n[w] += exi.instance(u).weight();
	}
	
	double logOdds = likelihoodRatio(n, xBar);
	return (logOdds > m_Cutoff) ? 0 : 1 ;
    }
    /**
     * Compute the log-likelihood ratio
     */
    private double likelihoodRatio(double[] n, double[] xBar){	
	double LLP = 0.0, LLN = 0.0;
	
	for (int x=0; x<m_Dimension; x++){
	    if(Double.isNaN(xBar[x])) continue; // All missing values
	    //if(Double.isNaN(xBar[x]) || (m_ParamsP[2*x] <= ZERO) 
	    //  || (m_ParamsN[2*x]<=ZERO)) 
	    //	continue; // All missing values
	    
	    //Log-likelihood for positive 
	    double w=m_ParamsP[2*x], m=m_ParamsP[2*x+1];
	    double llp = Math.log(w*n[x]+m_SgmSqP[x])
		+ n[x]*(m-xBar[x])*(m-xBar[x])/(w*n[x]+m_SgmSqP[x]);
	    LLP -= llp;
	    
	    //Log-likelihood for negative 
	    w=m_ParamsN[2*x]; m=m_ParamsN[2*x+1]; 
	    double lln = Math.log(w*n[x]+m_SgmSqN[x])
		+ n[x]*(m-xBar[x])*(m-xBar[x])/(w*n[x]+m_SgmSqN[x]);
	    LLN -= lln;

	    m_LkRatio[x] += llp - lln;
	}
	
	return LLP - LLN;
    }
    
    private void findCutOff(double[] pos, double[] neg){
	int[] pOrder = Utils.sort(pos),
	    nOrder = Utils.sort(neg);
	/*
	System.err.println("\n\n???Positive: ");
	for(int t=0; t<pOrder.length; t++)
	    System.err.print(t+":"+Utils.doubleToString(pos[pOrder[t]],0,2)+" ");
	System.err.println("\n\n???Negative: ");
	for(int t=0; t<nOrder.length; t++)
	    System.err.print(t+":"+Utils.doubleToString(neg[nOrder[t]],0,2)+" ");
	*/
	int pNum = pos.length, nNum = neg.length, count, p=0, n=0;	
	double total=(double)(pNum+nNum), 
	    fstAccu=0.0, sndAccu=(double)pNum, 
	    minEntropy=Double.MAX_VALUE, split; 
	double maxAccu = 0, minDistTo0 = Double.MAX_VALUE;
	
	// Skip continuous negatives	
	for(;(n<nNum)&&(pos[pOrder[0]]>=neg[nOrder[n]]); n++, fstAccu++);
	
	if(n>=nNum){ // totally seperate
	    m_Cutoff = (neg[nOrder[nNum-1]]+pos[pOrder[0]])/2.0;	
	    //m_Cutoff = neg[nOrder[nNum-1]];
	    return;  
	}	
	
	count=n;
	while((p<pNum)&&(n<nNum)){
	    // Compare the next in the two lists
	    if(pos[pOrder[p]]>=neg[nOrder[n]]){ // Neg has less log-odds
		fstAccu += 1.0;    
		split=neg[nOrder[n]];
		n++;	 
	    }
	    else{
		sndAccu -= 1.0;
		split=pos[pOrder[p]];
		p++;
	    }	    	  
	    count++;
	    /*
	    double entropy=0.0, cover=(double)count;
	    if(fstAccu>0.0)
		entropy -= fstAccu*Math.log(fstAccu/cover);
	    if(sndAccu>0.0)
		entropy -= sndAccu*Math.log(sndAccu/(total-cover));
	    
	    if(entropy < minEntropy){
		minEntropy = entropy;
		//find the next smallest
		//double next = neg[nOrder[n]];
		//if(pos[pOrder[p]]<neg[nOrder[n]])
		//    next = pos[pOrder[p]];	
		//m_Cutoff = (split+next)/2.0;
		m_Cutoff = split;
	    }
	    */
	    if((fstAccu+sndAccu > maxAccu) || 
	       ((fstAccu+sndAccu == maxAccu) && (Math.abs(split)<minDistTo0))){
		maxAccu = fstAccu+sndAccu;
		m_Cutoff = split;
		minDistTo0 = Math.abs(split);
	    }	    
	}		
    }
    
    /**
     * Returns an enumeration describing the available options
     * Valid options are: <p>
     *
     * -C Set whether or not use empirical log-odds cut-off instead of 0
     * (default: Not use) 
     *
     * -R <numOfRuns> Set the number of multiple runs needed for searching the MLE.
     * (default: 1)
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(1);
	newVector.addElement(new Option("\tSet whether or not use empirical\n"+
					"\tlog-odds cut-off instead of 0\n",
					"C", 0, "-C"));
	newVector.addElement(new Option("\tSet the number of multiple runs \n"+
					"\tneeded for searching the MLE.\n",
					"R", 1, "-R <numOfRuns>"));
	return newVector.elements();
    }
    
    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception{
	m_UseEmpiricalCutOff = Utils.getFlag('C', options);
	
	String runString = Utils.getOption('R', options);
	if (runString.length() != 0) 
	    m_Run = Integer.parseInt(runString);
	else 
	    m_Run = 1;	
    }
    
    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {
	
	String [] options = new String [3];
	int current = 0;
	options[current++] = "-C";
	options[current++] = "-R";
	options[current++] = ""+m_Run;
	while (current < options.length) 
	    options[current++] = "";
	return options;
    }

    /**
     * Gets a string describing the classifier.
     *
     * @return a string describing the classifer built.
     */
    public String toString(){
	StringBuffer text = new StringBuffer("\n\nTLDSimple:\n");
	double sgm, w, m;
	for (int x=0, y=0; x<m_Dimension; x++, y++){
	    if((x==m_ClassIndex) || (x==m_IdIndex))
		y++;
	    sgm = m_SgmSqP[x];
	    w=m_ParamsP[2*x]; 
	    m=m_ParamsP[2*x+1];
	    text.append("\n"+m_Attribute.attribute(y).name()+"\nPositive: "+
			"sigma^2="+sgm+", w="+w+", m="+m+"\n");
	    sgm = m_SgmSqN[x];
	    w=m_ParamsN[2*x]; 
	    m=m_ParamsN[2*x+1];
	    text.append("Negative: "+
			"sigma^2="+sgm+", w="+w+", m="+m+"\n");
	}

	return text.toString();
    }     
    
    /**
     * Main method for testing.
     *
     * @param args the options for the classifier
     */
    public static void main(String[] args) {	
	try {
	    System.out.println(MIEvaluation.evaluateModel(new TLDSimple(), args));
	} catch (Exception e) {
	    e.printStackTrace();
	    System.err.println(e.getMessage());
	}
    }
}

class TLDSimple_Optm extends Optimization{
    
    private double[] num;
    private double sSq;
    private double[] xBar;
    
    public void setNum(double[] n) {num = n;}
    public void setSgmSq(double s){sSq = s;}
    public void setXBar(double[] x){xBar = x;}
    
    /* 
     * Implement this procedure to evaluate objective
     * function to be minimized
     */
    protected double objectiveFunction(double[] x){
	int numExs = num.length;
	double NLL=0; // Negative Log-Likelihood
	
	double w=x[0], m=x[1];
	for(int j=0; j < numExs; j++){
	    
	    if(Double.isNaN(xBar[j])) continue; // All missing values
	    double bag=0;
	    bag += Math.log(w*num[j]+sSq);
	    if(Double.isNaN(bag)){
		System.out.println("???????????1: "+w+" "+m
				   +"|x-: "+xBar[j] + 
				   "|n: "+num[j] + "|S^2: "+sSq);
		//System.exit(1);
	    }
	    
	    bag += num[j]*(m-xBar[j])*(m-xBar[j])/(w*num[j]+sSq);	    	    
	    if(Double.isNaN(bag)){
		System.out.println("???????????2: "+w+" "+m
				   +"|x-: "+xBar[j] + 
				   "|n: "+num[j] + "|S^2: "+sSq);
		//System.exit(1);
	    }	    	       
	    
	    //if(bag<0) bag=0;
	    NLL += bag;
	}
	
	//System.out.println("???????????NLL:"+NLL);
	return NLL;
    }
    
    /* 
     * Subclass should implement this procedure to evaluate gradient
     * of the objective function
     */
    protected double[] evaluateGradient(double[] x){
	double[] g = new double[x.length];
	int numExs = num.length;
	
	double w=x[0],m=x[1];	
	double dw=0.0, dm=0.0;
	
	for(int j=0; j < numExs; j++){
	    
	    if(Double.isNaN(xBar[j])) continue; // All missing values	    
	    dw += num[j]/(w*num[j]+sSq) 
		- num[j]*num[j]*(m-xBar[j])*(m-xBar[j])/((w*num[j]+sSq)*(w*num[j]+sSq));
	    
	    dm += 2.0*num[j]*(m-xBar[j])/(w*num[j]+sSq);
	}
	
	g[0] = dw;
	g[1] = dm;
	return g;
    }
    
    /* 
     * Subclass should implement this procedure to evaluate second-order
     * gradient of the objective function
     */
    protected double[] evaluateHessian(double[] x, int index){
	double[] h = new double[x.length];

	// # of exemplars, # of dimensions
	// which dimension and which variable for 'index'
	int numExs = num.length;
	double w,m;
	// Take the 2nd-order derivative
	switch(index){	
	case 0: // w   
	    w=x[0];m=x[1];
	    
	    for(int j=0; j < numExs; j++){
		if(Double.isNaN(xBar[j])) continue; //All missing values
		
		h[0] += 2.0*Math.pow(num[j],3)*(m-xBar[j])*(m-xBar[j])/Math.pow(w*num[j]+sSq,3)
		    - num[j]*num[j]/((w*num[j]+sSq)*(w*num[j]+sSq));
		
		h[1] -= 2.0*(m-xBar[j])*num[j]*num[j]/((num[j]*w+sSq)*(num[j]*w+sSq));		
	    }
	    break;
	    
	case 1: // m
	    w=x[0];m=x[1];
	    
	    for(int j=0; j < numExs; j++){
		if(Double.isNaN(xBar[j])) continue; //All missing values
		
		h[0] -= 2.0*(m-xBar[j])*num[j]*num[j]/((num[j]*w+sSq)*(num[j]*w+sSq));
		
		h[1] += 2.0*num[j]/(w*num[j]+sSq);				
	    }
	}
	
	return h;
    }
}

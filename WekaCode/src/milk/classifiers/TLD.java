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
 * Two-Level Distribution approach, changes the 
 * starting value of the searching algorithm, supplement the cut-off
 * modification and check missing values. <p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.1 $
 */
public class TLD extends MIClassifier implements OptionHandler{

    /** The mean for each attribute of each positive exemplar */
    protected double[][] m_MeanP = null;

    /** The variance for each attribute of each positive exemplar */
    protected double[][] m_VarianceP = null;

    /** The mean for each attribute of each negative exemplar */
    protected double[][] m_MeanN = null;

    /** The variance for each attribute of each negative exemplar */
    protected double[][] m_VarianceN = null;

    /** The effective sum of weights of each positive exemplar in each dimension*/
    protected double[][] m_SumP = null;
    
    /** The effective sum of weights of each negative exemplar in each dimension*/
    protected double[][] m_SumN = null;
    
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
    static public double ZERO = 1.0e-6;   

    protected int m_Run = 1;

    protected long m_Seed = 1;

    protected double m_Cutoff;

    protected boolean m_UseEmpiricalCutOff = false;    
    
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
	
	Exemplars pos = new Exemplars(exs, 0), neg = new Exemplars(exs, 0);
	
	for(int u=0; u<numegs; u++){
	    Exemplar example = exs.exemplar(u);
	    if(example.classValue() == 0)
		pos.add(example);
	    else
		neg.add(example);
	}
	
	int pnum = pos.numExemplars(), nnum = neg.numExemplars();	
	
	m_MeanP = new double[pnum][m_Dimension];
	m_VarianceP = new double[pnum][m_Dimension];
	m_SumP = new double[pnum][m_Dimension];
	m_MeanN = new double[nnum][m_Dimension];
	m_VarianceN = new double[nnum][m_Dimension];
	m_SumN = new double[nnum][m_Dimension];
	m_ParamsP = new double[4*m_Dimension];
	m_ParamsN = new double[4*m_Dimension];
	
	// Estimation of the parameters: as the start value for search
	double[] pSumVal=new double[m_Dimension], // for m 
	    nSumVal=new double[m_Dimension]; 
	double[] maxVarsP=new double[m_Dimension], // for a
	    maxVarsN=new double[m_Dimension]; 
	// Mean of sample variances: for b, b=a/E(\sigma^2)+2
	double[] varMeanP = new double[m_Dimension],
	    varMeanN = new double[m_Dimension]; 
	// Variances of sample means: for w, w=E[var(\mu)]/E[\sigma^2]
	double[] meanVarP = new double[m_Dimension],
	    meanVarN = new double[m_Dimension];
	// number of exemplars without all values missing
	double[] numExsP = new double[m_Dimension],
	    numExsN = new double[m_Dimension];

	// Extract metadata fro both positive and negative bags
	for(int v=0; v < pnum; v++){
	    Exemplar px = pos.exemplar(v);
	    m_MeanP[v] = px.meanOrMode();
	    m_VarianceP[v] = px.variance();
	    Instances pxi =  px.getInstances();
	    
	    for (int w=0,t=0; w < m_Dimension; w++,t++){		
		if((t==m_ClassIndex) || (t==m_IdIndex))
		    t++;		

		if(!Double.isNaN(m_MeanP[v][w])){
		    for(int u=0;u<pxi.numInstances();u++){
			Instance ins = pxi.instance(u);			
			if(!ins.isMissing(t))
			    m_SumP[v][w] += ins.weight();			   
		    }   
		    numExsP[w]++;  
		    pSumVal[w] += m_MeanP[v][w];
		    meanVarP[w] += m_MeanP[v][w]*m_MeanP[v][w];    
		    if(maxVarsP[w] < m_VarianceP[v][w])
			maxVarsP[w] = m_VarianceP[v][w];
		    varMeanP[w] += m_VarianceP[v][w];
		    m_VarianceP[v][w] *= (m_SumP[v][w]-1.0);
		    if(m_VarianceP[v][w] < 0.0)
			m_VarianceP[v][w] = 0.0;
		}
	    }
	}
	
	for(int v=0; v < nnum; v++){
	    Exemplar nx = neg.exemplar(v);
	    m_MeanN[v] = nx.meanOrMode();
	    m_VarianceN[v] = nx.variance();
	    Instances nxi =  nx.getInstances();
	    
	    for (int w=0,t=0; w < m_Dimension; w++,t++){		
		if((t==m_ClassIndex) || (t==m_IdIndex))
		    t++;		
		
		if(!Double.isNaN(m_MeanN[v][w])){
		    for(int u=0;u<nxi.numInstances();u++)
			if(!nxi.instance(u).isMissing(t))
			    m_SumN[v][w] += nxi.instance(u).weight();
		    numExsN[w]++; 	
		    nSumVal[w] += m_MeanN[v][w];
		    meanVarN[w] += m_MeanN[v][w]*m_MeanN[v][w]; 
		    if(maxVarsN[w] < m_VarianceN[v][w])
			maxVarsN[w] = m_VarianceN[v][w];
		    varMeanN[w] += m_VarianceN[v][w];
		    m_VarianceN[v][w] *= (m_SumN[v][w]-1.0);
		    if(m_VarianceN[v][w] < 0.0)
			m_VarianceN[v][w] = 0.0;
		}
	    }
	}
	
	for(int w=0; w<m_Dimension; w++){
	    pSumVal[w] /= numExsP[w];
	    nSumVal[w] /= numExsN[w];
	    if(numExsP[w]>1)
		meanVarP[w] = meanVarP[w]/(numExsP[w]-1.0) 
		    - pSumVal[w]*numExsP[w]/(numExsP[w]-1.0);
	    if(numExsN[w]>1)
		meanVarN[w] = meanVarN[w]/(numExsN[w]-1.0) 
		    - nSumVal[w]*numExsN[w]/(numExsN[w]-1.0);
	    varMeanP[w] /= numExsP[w];
	    varMeanN[w] /= numExsN[w];
	}
	
	//Bounds and parameter values for each run
	double[][] bounds = new double[2][4];
	double[] pThisParam = new double[4], 
	    nThisParam = new double[4];
	
	// Initial values for parameters
	double a, b, w, m;
	
	// Optimize for one dimension
	for (int x=0; x < m_Dimension; x++){     
	    System.err.println("\n\n!!!!!!!!!!!!!!!!!!!!!!???Dimension #"+x);
	    
	    // Positive examplars: first run
	    a = (maxVarsP[x]>ZERO) ? maxVarsP[x]:1.0; 
	    b = a/varMeanP[x]+2.0; // a/(b-2) = E(\sigma^2)
	    w = meanVarP[x]/varMeanP[x]; // E[var(\mu)] = w*E[\sigma^2]
	    if(w<=ZERO) w=1.0;
	    m = pSumVal[x]; 	  
	    pThisParam[0] = a;    // a
	    pThisParam[1] = b;  // b
	    pThisParam[2] = w;  // w
	    pThisParam[3] = m;  // m
	    
	    // Negative examplars: first run
	    a = (maxVarsN[x]>ZERO) ? maxVarsN[x]:1.0; 
	    b = a/varMeanN[x]+2.0; // a/(b-2) = E(\sigma^2)
	    w = meanVarN[x]/varMeanN[x]; // E[var(\mu)] = w*E[\sigma^2]
	    if(w<=ZERO) w=1.0;
	    m = nSumVal[x]; 	  
	    nThisParam[0] = a;    // a
	    nThisParam[1] = b;  // b
	    nThisParam[2] = w;  // w
	    nThisParam[3] = m;  // m
	    
	    // Bound constraints
	    bounds[0][0] = ZERO; // a > 0
	    bounds[0][1] = 2.0+ZERO;  // b > 2 
	    bounds[0][2] = ZERO; // w > 0
	    bounds[0][3] = Double.NaN;
	    
	    for(int t=0; t<4; t++){
		bounds[1][t] = Double.NaN;
		m_ParamsP[4*x+t] = pThisParam[t];	
		m_ParamsN[4*x+t] = nThisParam[t];
	    }
	    double pminVal=Double.MAX_VALUE, nminVal=Double.MAX_VALUE;
	    Random whichEx = new Random(m_Seed); 
	    TLD_Optm pOp=null, nOp=null;	
	    boolean isRunValid = true;
	    double[] sumP=new double[pnum], meanP=new double[pnum],
		varP=new double[pnum];
	    double[] sumN=new double[nnum], meanN=new double[nnum],
		varN=new double[nnum];
	    
	    // One dimension
	    for(int p=0; p<pnum; p++){
		sumP[p] = m_SumP[p][x];
		meanP[p] = m_MeanP[p][x];
		varP[p] = m_VarianceP[p][x];
	    }
	    for(int q=0; q<nnum; q++){
		sumN[q] = m_SumN[q][x];
		meanN[q] = m_MeanN[q][x];
		varN[q] = m_VarianceN[q][x];
	    }
	    
	    for(int y=0; y<m_Run;){
		System.err.println("\n\n!!!!!!!!!!!!!!!!!!!!!!???Run #"+y);
		double thisMin;
		
		System.err.println("\nPositive exemplars");
		pOp = new TLD_Optm();
		pOp.setNum(sumP);
		pOp.setSSquare(varP);
		pOp.setXBar(meanP);

		pThisParam = pOp.findArgmin(pThisParam, bounds);
		while(pThisParam==null){
		    pThisParam = pOp.getVarbValues();		    
		    System.err.println("!!! 200 iterations finished, not enough!");
		    pThisParam = pOp.findArgmin(pThisParam, bounds);
		}	
		
		thisMin = pOp.getMinFunction();
		if(!Double.isNaN(thisMin) && (thisMin<pminVal)){
		    pminVal = thisMin;
		    for(int z=0; z<4; z++)
			m_ParamsP[4*x+z] = pThisParam[z];
		}
		
		if(Double.isNaN(thisMin)){
		    pThisParam = new double[4];
		    isRunValid =false;
		}
		
		System.err.println("\nNegative exemplars");
		nOp = new TLD_Optm();
		nOp.setNum(sumN);
		nOp.setSSquare(varN);
		nOp.setXBar(meanN);

		nThisParam = nOp.findArgmin(nThisParam, bounds);
		while(nThisParam==null){
		    nThisParam = nOp.getVarbValues();
		    System.err.println("!!! 200 iterations finished, not enough!");
		    nThisParam = nOp.findArgmin(nThisParam, bounds);
		}	
		thisMin = nOp.getMinFunction();
		if(!Double.isNaN(thisMin) && (thisMin<nminVal)){
		    nminVal = thisMin;
		    for(int z=0; z<4; z++)
			m_ParamsN[4*x+z] = nThisParam[z];     
		}
		
		if(Double.isNaN(thisMin)){
		    nThisParam = new double[4];
		    isRunValid =false;
		}
		
		if(!isRunValid){ y--; isRunValid=true; } 		
		
		if(++y<m_Run){
		    // Change the initial parameters and restart	   	    
		    int pone = whichEx.nextInt(pnum), // Randomly pick one pos. exmpl.
			none = whichEx.nextInt(nnum);
		    
		    // Positive exemplars: next run 
		    while((m_SumP[pone][x]<=1.0)||Double.isNaN(m_MeanP[pone][x]))
			pone = whichEx.nextInt(pnum);
		    
		    a = m_VarianceP[pone][x]/(m_SumP[pone][x]-1.0); 		
		    if(a<=ZERO) a=m_ParamsN[4*x]; // Change to negative params
		    m = m_MeanP[pone][x];
		    double sq = (m-m_ParamsP[4*x+3])*(m-m_ParamsP[4*x+3]);
		    
		    b = a*m_ParamsP[4*x+2]/sq+2.0; // b=a/Var+2, assuming Var=Sq/w'
		    if((b<=ZERO) || Double.isNaN(b) || Double.isInfinite(b))
			b=m_ParamsN[4*x+1];
		    
		    w = sq*(m_ParamsP[4*x+1]-2.0)/m_ParamsP[4*x];//w=Sq/Var, assuming Var=a'/(b'-2)
		    if((w<=ZERO) || Double.isNaN(w) || Double.isInfinite(w))
			w=m_ParamsN[4*x+2];
		
		    pThisParam[0] = a;    // a
		    pThisParam[1] = b;  // b
		    pThisParam[2] = w;  // w
		    pThisParam[3] = m;  // m	    
		    
		    // Negative exemplars: next run 
		    while((m_SumN[none][x]<=1.0)||Double.isNaN(m_MeanN[none][x]))
			none = whichEx.nextInt(nnum);	    
		    
		    a = m_VarianceN[none][x]/(m_SumN[none][x]-1.0);	
		    if(a<=ZERO) a=m_ParamsP[4*x];       
		    m = m_MeanN[none][x];
		    sq = (m-m_ParamsN[4*x+3])*(m-m_ParamsN[4*x+3]);
		    
		    b = a*m_ParamsN[4*x+2]/sq+2.0; // b=a/Var+2, assuming Var=Sq/w'
		    if((b<=ZERO) || Double.isNaN(b) || Double.isInfinite(b))
			b=m_ParamsP[4*x+1];
		    
		    w = sq*(m_ParamsN[4*x+1]-2.0)/m_ParamsN[4*x];//w=Sq/Var, assuming Var=a'/(b'-2)
		    if((w<=ZERO) || Double.isNaN(w) || Double.isInfinite(w))
			w=m_ParamsP[4*x+2];
		    
		    nThisParam[0] = a;    // a
		    nThisParam[1] = b;  // b
		    nThisParam[2] = w;  // w
		    nThisParam[3] = m;  // m	    		
		}
	    }	    	    	    
	}
	
	for (int x=0, y=0; x<m_Dimension; x++, y++){
	    if((x==exs.classIndex()) || (x==exs.idIndex()))
		y++;
	    a=m_ParamsP[4*x]; b=m_ParamsP[4*x+1]; 
	    w=m_ParamsP[4*x+2]; m=m_ParamsP[4*x+3];
	    System.err.println("\n\n???Positive: ( "+exs.attribute(y)+
			       "): a="+a+", b="+b+", w="+w+", m="+m);
	    
	    a=m_ParamsN[4*x]; b=m_ParamsN[4*x+1]; 
	    w=m_ParamsN[4*x+2]; m=m_ParamsN[4*x+3];
	    System.err.println("???Negative: ("+exs.attribute(y)+
			       "): a="+a+", b="+b+", w="+w+", m="+m);
	}

	if(m_UseEmpiricalCutOff){	
	    // Find the empirical cut-off
	    double[] pLogOdds=new double[pnum], nLogOdds=new double[nnum];  
	    for(int p=0; p<pnum; p++)
		pLogOdds[p] = 
		    likelihoodRatio(m_SumP[p], m_MeanP[p], m_VarianceP[p]);
	    
	    for(int q=0; q<nnum; q++)
		nLogOdds[q] = 
		    likelihoodRatio(m_SumN[q], m_MeanN[q], m_VarianceN[q]);
	    
	    // Update m_Cutoff
	    findCutOff(pLogOdds, nLogOdds);
	}
	else
	    m_Cutoff = -Math.log((double)pnum/(double)nnum);
     
	System.err.println("???Cut-off="+m_Cutoff);
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
	double[] n = new double[m_Dimension], xBar = ex.meanOrMode(), 
	    sSq = ex.variance();
	
	for (int w=0, t=0; w < m_Dimension; w++, t++){
	    if((t==m_ClassIndex) || (t==m_IdIndex))
		t++;	
	    for(int u=0;u<exi.numInstances();u++)
		if(!exi.instance(u).isMissing(t))
		    n[w] += exi.instance(u).weight();
	    
	    sSq[w] = sSq[w]*(n[w]-1.0);
	    if(sSq[w] <= 0.0)
		sSq[w] = 0.0;
	}
	
	double logOdds = likelihoodRatio(n, xBar, sSq);
	return (logOdds > m_Cutoff) ? 0 : 1 ;
    }
    
    private double likelihoodRatio(double[] n, double[] xBar, double[] sSq){	
	double LLP = 0.0, LLN = 0.0;
	
	for (int x=0; x<m_Dimension; x++){
	    if(Double.isNaN(xBar[x])) continue; // All missing values
	    
	    int halfN = ((int)n[x])/2;	
	    //Log-likelihood for positive 
	    double a=m_ParamsP[4*x], b=m_ParamsP[4*x+1], 
		w=m_ParamsP[4*x+2], m=m_ParamsP[4*x+3];
	    LLP += 0.5*b*Math.log(a) + 0.5*(b+n[x]-1.0)*Math.log(1.0+n[x]*w)
		- 0.5*(b+n[x])*Math.log((1.0+n[x]*w)*(a+sSq[x])+
					n[x]*(xBar[x]-m)*(xBar[x]-m))
		- 0.5*n[x]*Math.log(Math.PI);
	    for(int y=1; y<=halfN; y++)
		LLP += Math.log(b/2.0+n[x]/2.0-(double)y);
	    
	    if(n[x]/2.0 > halfN) // n is odd
		LLP += TLD_Optm.diffLnGamma(b/2.0);
	    
	    //Log-likelihood for negative 
	    a=m_ParamsN[4*x];
	    b=m_ParamsN[4*x+1]; 
	    w=m_ParamsN[4*x+2];
	    m=m_ParamsN[4*x+3];
	    LLN += 0.5*b*Math.log(a) + 0.5*(b+n[x]-1.0)*Math.log(1.0+n[x]*w)
		- 0.5*(b+n[x])*Math.log((1.0+n[x]*w)*(a+sSq[x])+
					n[x]*(xBar[x]-m)*(xBar[x]-m))
		- 0.5*n[x]*Math.log(Math.PI);
	    for(int y=1; y<=halfN; y++)
		LLN += Math.log(b/2.0+n[x]/2.0-(double)y);	
	    
	    if(n[x]/2.0 > halfN) // n is odd
		LLN += TLD_Optm.diffLnGamma(b/2.0);   
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

    /*private void findCutOff(double[] pos, double[] neg){
	int[] pOrder = Utils.sort(pos),
	    nOrder = Utils.sort(neg);
	
	int pNum = pos.length, nNum = neg.length, count, p=0, n=0;	
	double total=(double)(pNum+nNum), 
	    fstAccu=0.0, sndAccu=(double)pNum, 
	    minEntropy=Double.MAX_VALUE, split; 

	// Skip continuous negatives	
	for(;(n<nNum)&&(pos[pOrder[0]]>neg[nOrder[n]]); n++, fstAccu++);
	
	if(n>=nNum){ // totally seperate
	    m_Cutoff = neg[nOrder[nNum-1]];
	    return;
	}	

	count=n;
	while((p<pNum)&&(n<nNum)){
	    // Compare the next in the two lists
	    if(pos[pOrder[p]]>neg[nOrder[n]]){ // Neg has less log-odds
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

	    double entropy=0.0, cover=(double)count;
	    if(fstAccu>0.0)
		entropy -= fstAccu*Math.log(fstAccu/cover);
	    if(sndAccu>0.0)
		entropy -= sndAccu*Math.log(sndAccu/(total-cover));
	    
	    if(entropy < minEntropy){
		minEntropy = entropy;
		m_Cutoff = split;
	    }
	}	
    }
    */

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
     * Main method for testing.
     *
     * @param args the options for the classifier
     */
    public static void main(String[] args) {	
	try {
	    System.out.println(MIEvaluation.evaluateModel(new TLD(), args));
	} catch (Exception e) {
	    e.printStackTrace();
	    System.err.println(e.getMessage());
	}
    }
}

class TLD_Optm extends Optimization{
    
    private double[] num;
    private double[] sSq;
    private double[] xBar;
    
    public void setNum(double[] n) {num = n;}
    public void setSSquare(double[] s){sSq = s;}
    public void setXBar(double[] x){xBar = x;}
    
    /**
     * Compute Ln[Gamma(b+0.5)] - Ln[Gamma(b)]
     *
     * @param b the value in the above formula
     * @return the result
     */    
    public static double diffLnGamma(double b){
	double[] coef= {76.18009172947146, -86.50532032941677,
			24.01409824083091, -1.231739572450155, 
			0.1208650973866179e-2, -0.5395239384953e-5};
	double rt = -0.5;
	rt += (b+1.0)*Math.log(b+6.0) - (b+0.5)*Math.log(b+5.5);
	double series1=1.000000000190015, series2=1.000000000190015;
	for(int i=0; i<6; i++){
	    series1 += coef[i]/(b+1.5+(double)i);
	    series2 += coef[i]/(b+1.0+(double)i);
	}
	
	rt += Math.log(series1*b)-Math.log(series2*(b+0.5));
	return rt;
    }
    
    /**
     * Compute dLn[Gamma(x+0.5)]/dx - dLn[Gamma(x)]/dx
     *
     * @param x the value in the above formula
     * @return the result
     */    
    protected double diffFstDervLnGamma(double x){
	double rt=0, series=1.0;// Just make it >0
	for(int i=0;series>=m_Zero*1e-3;i++){
	    series = 0.5/((x+(double)i)*(x+(double)i+0.5));
	    rt += series;
	}
	return rt;
    }
    
    /**
     * Compute {Ln[Gamma(x+0.5)]}'' - {Ln[Gamma(x)]}''
     *
     * @param x the value in the above formula
     * @return the result
     */    
    protected double diffSndDervLnGamma(double x){
	double rt=0, series=1.0;// Just make it >0
	for(int i=0;series>=m_Zero*1e-3;i++){
	    series = (x+(double)i+0.25)/
		((x+(double)i)*(x+(double)i)*(x+(double)i+0.5)*(x+(double)i+0.5));
	    rt -= series;
	}
	return rt;
    }
    
    /* 
     * Implement this procedure to evaluate objective
     * function to be minimized
     */
    protected double objectiveFunction(double[] x){
	int numExs = num.length;
	double NLL = 0; // Negative Log-Likelihood
	
	double a=x[0], b=x[1], w=x[2], m=x[3];
	for(int j=0; j < numExs; j++){
	    
	    if(Double.isNaN(xBar[j])) continue; // All missing values
	    
	    NLL += 0.5*(b+num[j])*
		Math.log((1.0+num[j]*w)*(a+sSq[j]) + 
			 num[j]*(xBar[j]-m)*(xBar[j]-m));	    
	    
	    if(Double.isNaN(NLL)){
		System.err.println("???????????1: "+a+" "+b+" "+w+" "+m
				   +"|x-: "+xBar[j] + 
				   "|n: "+num[j] + "|S^2: "+sSq[j]);
		System.exit(1);
	    }
	    
	    // Doesn't affect optimization
	    //NLL += 0.5*num[j]*Math.log(Math.PI);		
	    
	    NLL -= 0.5*(b+num[j]-1.0)*Math.log(1.0+num[j]*w);
	    
	    
	    if(Double.isNaN(NLL)){
		System.err.println("???????????2: "+a+" "+b+" "+w+" "+m
				   +"|x-: "+xBar[j] + 
				   "|n: "+num[j] + "|S^2: "+sSq[j]);
		System.exit(1);
	    }
	    
	    int halfNum = ((int)num[j])/2;
	    for(int z=1; z<=halfNum; z++)
		NLL -= Math.log(0.5*b+0.5*num[j]-(double)z);
	    
	    if(0.5*num[j] > halfNum) // num[j] is odd
		NLL -= diffLnGamma(0.5*b);
	    
	    if(Double.isNaN(NLL)){
		System.err.println("???????????3: "+a+" "+b+" "+w+" "+m
				   +"|x-: "+xBar[j] + 
				   "|n: "+num[j] + "|S^2: "+sSq[j]);
		System.exit(1);
	    }				
	    
	    NLL -= 0.5*Math.log(a)*b;
	    if(Double.isNaN(NLL)){
		System.err.println("???????????4:"+a+" "+b+" "+w+" "+m);
		System.exit(1);
	    }	    
	}
	
	System.err.println("?????????????5: "+NLL);
	if(Double.isNaN(NLL))	   
	    System.exit(1);
	
	return NLL;
    }
    
    /* 
     * Subclass should implement this procedure to evaluate gradient
     * of the objective function
     */
    protected double[] evaluateGradient(double[] x){
	double[] g = new double[x.length];
	int numExs = num.length;
	
	double a=x[0],b=x[1],w=x[2],m=x[3];
	
	double da=0.0, db=0.0, dw=0.0, dm=0.0; 
	for(int j=0; j < numExs; j++){
	    
	    if(Double.isNaN(xBar[j])) continue; // All missing values
	    
	    double denorm = (1.0+num[j]*w)*(a+sSq[j]) + 
		num[j]*(xBar[j]-m)*(xBar[j]-m);
	    
	    da += 0.5*(b+num[j])*(1.0+num[j]*w)/denorm-0.5*b/a;
	    
	    db += 0.5*Math.log(denorm) 
		- 0.5*Math.log(1.0+num[j]*w)
		- 0.5*Math.log(a);
	    
	    int halfNum = ((int)num[j])/2;
	    for(int z=1; z<=halfNum; z++)
		db -= 1.0/(b+num[j]-2.0*(double)z);		
	    if(num[j]/2.0 > halfNum) // num[j] is odd
		db -= 0.5*diffFstDervLnGamma(0.5*b);		
	    
	    dw += 0.5*(b+num[j])*(a+sSq[j])*num[j]/denorm -
		0.5*(b+num[j]-1.0)*num[j]/(1.0+num[j]*w);
	    
	    dm += num[j]*(b+num[j])*(m-xBar[j])/denorm;
	}
	
	g[0] = da;
	g[1] = db;
	g[2] = dw;
	g[3] = dm;
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
	double a,b,w,m;
	// Take the 2nd-order derivative
	switch(index){
	case 0:  // a	   
	    a=x[0];b=x[1];w=x[2];m=x[3];
	   
	    for(int j=0; j < numExs; j++){
		if(Double.isNaN(xBar[j])) continue; //All missing values
		double denorm = (1.0+num[j]*w)*(a+sSq[j]) + 
		    num[j]*(xBar[j]-m)*(xBar[j]-m);
		
		h[0] += 0.5*b/(a*a) 
		    - 0.5*(b+num[j])*(1.0+num[j]*w)*(1.0+num[j]*w)
		    /(denorm*denorm);
		
		h[1] += 0.5*(1.0+num[j]*w)/denorm - 0.5/a;
		
		h[2] += 0.5*num[j]*num[j]*(b+num[j])*
		    (xBar[j]-m)*(xBar[j]-m)/(denorm*denorm);
		
		h[3] -= num[j]*(b+num[j])*(m-xBar[j])
		    *(1.0+num[j]*w)/(denorm*denorm);
	    }
	    break;
	    
	case 1: // b      
	    a=x[0];b=x[1];w=x[2];m=x[3];
	   
	    for(int j=0; j < numExs; j++){
		if(Double.isNaN(xBar[j])) continue; //All missing values
		double denorm = (1.0+num[j]*w)*(a+sSq[j]) + 
		    num[j]*(xBar[j]-m)*(xBar[j]-m);
		
		h[0] += 0.5*(1.0+num[j]*w)/denorm - 0.5/a;
		
		int halfNum = ((int)num[j])/2;
		for(int z=1; z<=halfNum; z++)
		    h[1] += 
			1.0/((b+num[j]-2.0*(double)z)*(b+num[j]-2.0*(double)z));
		if(num[j]/2.0 > halfNum) // num[j] is odd
		    h[1] -= 0.25*diffSndDervLnGamma(0.5*b); 
		
		h[2] += 0.5*(a+sSq[j])*num[j]/denorm -
		    0.5*num[j]/(1.0+num[j]*w);
		
		h[3] += num[j]*(m-xBar[j])/denorm;
	    }
	    break;
	    
	case 2: // w   
	    a=x[0];b=x[1];w=x[2];m=x[3];
	    
	    for(int j=0; j < numExs; j++){
		if(Double.isNaN(xBar[j])) continue; //All missing values
		double denorm = (1.0+num[j]*w)*(a+sSq[j]) + 
		    num[j]*(xBar[j]-m)*(xBar[j]-m);
		
		h[0] += 0.5*num[j]*num[j]*(b+num[j])*
		    (xBar[j]-m)*(xBar[j]-m)/(denorm*denorm);
		
		h[1] += 0.5*(a+sSq[j])*num[j]/denorm -
		    0.5*num[j]/(1.0+num[j]*w);
		
		h[2] += 0.5*(b+num[j]-1.0)*num[j]*num[j]/
		    ((1.0+num[j]*w)*(1.0+num[j]*w)) -
		    0.5*(b+num[j])*(a+sSq[j])*(a+sSq[j])*
		    num[j]*num[j]/(denorm*denorm);
		
		h[3] -= num[j]*num[j]*(b+num[j])*
		    (m-xBar[j])*(a+sSq[j])/(denorm*denorm);
	    }
	    break;
	    
	case 3: // m
	    a=x[0];b=x[1];w=x[2];m=x[3];
	    
	    for(int j=0; j < numExs; j++){
		if(Double.isNaN(xBar[j])) continue; //All missing values
		double denorm = (1.0+num[j]*w)*(a+sSq[j]) + 
		    num[j]*(xBar[j]-m)*(xBar[j]-m);
		
		h[0] -= num[j]*(b+num[j])*(m-xBar[j])
		    *(1.0+num[j]*w)/(denorm*denorm);
		
		h[1] += num[j]*(m-xBar[j])/denorm;
		
		h[2] -= num[j]*num[j]*(b+num[j])*
		    (m-xBar[j])*(a+sSq[j])/(denorm*denorm);
		
		h[3] += num[j]*(b+num[j])*
		    ((1.0+num[j]*w)*(a+sSq[j])-
		     num[j]*(m-xBar[j])*(m-xBar[j]))
		    /(denorm*denorm);
	    }
	}
	
	return h;
    }
}

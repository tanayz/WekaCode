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
 *    MIDataSample.java
 *
 */
package milk.data;

import weka.core.Utils;
import java.lang.*;
import java.io.*;
import java.util.Random;
/**
 * Generate n-dimensional 2-class MI data using instance-based approach
 * sample version
 */
public class MIDataSample{   
    public static void main(String[] args) {  	
	try{
	    MIDataSample data = new MIDataSample();
	    data.generateData(args);
	}catch (Exception ex) {
	    System.err.println(ex.getMessage());
	    ex.printStackTrace();
	}
    }
    
    public void generateData(String[] args) throws Exception{
	try{  
	    int index=0;
	    String seedString = Utils.getOption('s', args);
	    String fileName = args[index++];
	    int numEx = Integer.parseInt(args[index++]),
		numAtts = Integer.parseInt(args[index++]);
	    long seed = 1;
	    double[] beta = new double[numAtts];
	    for(int i=0; i<numAtts; i++)
		beta[i] = Double.parseDouble(args[index++]);
	    if(seedString != "")
		seed = Long.parseLong(seedString);

	    Random ran = new Random(seed);	    
	    double prob[]=new double[numEx], x[][]=new double[numEx][numAtts],
		range[][] = new double[numEx][numAtts];
	    int cla[]=new int[numEx], labels[][]=new int[numEx][];
	    double wrong0=0, wrong1=0, wrong=0;
	    double[][][] data = new double[numEx][][];

	    for(int y=0; y < numEx; y++){
		int size = (int)(ran.nextDouble()*20)+1;
		data[y] = new double[size][numAtts];
		labels[y] = new int[size];
		for(int z=0; z<numAtts; z++){
		    // one centroid
		    x[y][z]=ran.nextDouble()*10-5; // A bag in [-5, 5) 
		    range[y][z]=ran.nextDouble()*4+2;
		    for(int a=0; a<size; a++)
			data[y][a][z] = nextPoint(x[y][z], ran, range[y][z]);
		}
		
		prob[y] = exemplarPosterior(data[y], beta, ran, labels[y]);
		cla[y] = (ran.nextDouble() > prob[y])? 0:1;
 
		if(prob[y]>=0.5){
		    wrong += (1-prob[y]);
		    if(cla[y]==0)
		    wrong0++;
		}
		else{
		    wrong += prob[y];
		    if(cla[y]==1)
			wrong1++;
		}
	    }
	    
	    PrintWriter pw = new PrintWriter(new FileWriter(fileName+".arff"));
	    pw.println("% Wrong 0 = "+wrong0+", Wrong 1 = "+wrong1+
		       ", Best prediction error = "+Utils.doubleToString((wrong0+wrong1)*100.0/numEx,5,2)+"%"
		       +" asymptotic Bayes error = "+Utils.doubleToString(wrong*100.0/numEx,7,4)+"%");
	    pw.print("% Log-Odds = ");
	    for(int k=0; k<numAtts; k++){
		if(k == numAtts-1)
		    pw.println(beta[k]+"*x["+k+"]");
		else
		    pw.print(beta[k]+"*x["+k+"] + ");
	    }
	    pw.println("%seed = "+seed+"\n");
	    pw.println("@relation "+fileName+"\n");
	    pw.print("@attribute examplar {");
	    
	    StringBuffer[] bagID = new StringBuffer[numEx]; 
	    for(int j=0; j<numEx; j++){
		bagID[j] = new StringBuffer(j+"_");
		
		// range
		for(int k=0; k<numAtts; k++)
		    bagID[j].append(Utils.doubleToString(range[j][k],5,4)+"_");
		
		// centroid
		for(int k=0; k<numAtts-1; k++)
		    bagID[j].append(Utils.doubleToString(x[j][k],3,2)+"_");
		bagID[j].append(Utils.doubleToString(x[j][numAtts-1],3,2));
	    }
	    
	    for(int j=0; j<numEx; j++){
		pw.print(bagID[j].toString());
		if(j == (numEx-1))
		    pw.println("}");
		else
		    pw.print(",");
	    }
	    
	    for(int i=1; i<=numAtts; i++)
		pw.println("@attribute X"+i+" numeric");
	    pw.println("@attribute class {0,1}");
	    pw.println("\n@data");
	    
	    for(int y=0; y < numEx; y++){        
		for(int a=0; a<data[y].length; a++){
		    pw.print(bagID[y].toString()+", ");
		    for(int b=0; b<numAtts; b++){
			pw.print(data[y][a][b]+", ");
		    }
		    //pw.println(labels[y][a]);
		    pw.println(cla[y]);
		}		
	    }
	    pw.close();
	}catch(Exception e){
	    e.printStackTrace();
	    throw new Exception("MIData: java MI** <filename> <numExamplers> "+
				"<numAttributes> "+"<coefficient of Attribute1>"
				+" <coefficient of Attribute2> ...\n\n"+
				e.getMessage());
	}
    }
    
    // Returns a linear logistic posterior
    protected double exemplarPosterior(double[][] x, double[] beta, Random ran, int[] c) 
	throws Exception{
	double prob=0;
	
	/*
	// sum of probs.
	for(int z=0; z<x.length; z++){	    
	    double logodds=0; // log-odds of this instance to be positive
	    for(int y=0; y<x[z].length; y++)
		logodds += x[z][y]*beta[y];	    
	    double p = Math.exp(logodds)/(1.0+Math.exp(logodds));// prob of 1
	    c[z] = (ran.nextDouble() > p)? 0:1;
	    prob += p;
	} 
	prob /= (double)x.length;
	
	return prob;	
	*/
	
	///*
	  // sum of log-odds
	for(int z=0; z<x.length; z++){	    
	    double logodds=0; // Prob. of this bag to be positive
	    for(int y=0; y<x[z].length; y++)
		logodds += x[z][y]*beta[y];	
	    prob += logodds;              
	}
	prob /= (double)x.length;
	prob = Math.exp(prob)/(1+Math.exp(prob));
	return prob;	
	//*/

	/*
	// test1: generate class label first then voting
	for(int z=0; z<x.length; z++){ 
	    double logodds=0; // log-odds of this instance to be positive
	    for(int y=0; y<x[z].length; y++)
		logodds += x[z][y]*beta[y];	
	    double p=Math.exp(logodds)/(1.0+Math.exp(logodds));
	    if(ran.nextDouble()>p) // select 1 as base class
		prob -= 1;
	    else
		prob += 1;
	}
	return (prob>0)?1:0; // return the prob.
	*/

	/*
	// test2: use the centroid as a representative
	double logodds=0;
	for(int y=0; y<c.length; y++)
	    logodds += c[y]*beta[y];	    
	prob = Math.exp(logodds)/(1.0+Math.exp(logodds));
	return prob;
	*/
    }
    
    // Generate random variates in a range of [x0-range/2, x0+range/2), 
    // with normalized triangle distribution in [-5, 5): 0.2-0.04|x|
    protected double nextPoint(double x0, Random ran, double range) throws Exception{
	double half = range/2;
	if(x0 >= half){
	    double point = ran.nextDouble()*range+(x0-half);
	    while(point >= 5)
		point = ran.nextDouble()*range+(x0-half);
	    if(point <= x0) // Keep it
		return point;
	    else if(ran.nextDouble()*(0.2-0.04*x0) > 0.2-0.04*point)
		return 2.0*x0-point;
	    else
		return point;       
	}
	else if(x0 <= -half){
	    double point = ran.nextDouble()*range+(x0-half);
	    while(point <= -5)
		point = ran.nextDouble()*range+(x0-half);
	    if(point >= x0) // Keep it
		return point;
	    else if(ran.nextDouble()*(0.2+0.04*x0) > 0.2+0.04*point)
		return 2.0*x0-point;
	    else
		return point;    
	}
	else{
	    double total=-0.04*x0*x0-0.04*half*half+0.4*half, 
		left = -0.02*x0*x0-0.2*x0+0.04*half*x0-0.02*half*half+0.2*half;
	    
	    if(ran.nextDouble()*total <= left){ // Left side
		double point = (ran.nextDouble()-1)*(half-x0);
		if(point >= (x0-half)/2.0) // Keep it
		    return point;
		else if(ran.nextDouble()*(0.2+0.02*(x0-half)) > 0.2+0.04*point)
		    return x0-half-point;
		else
		    return point;       
	    }
	    else{ // Right side
		double point = ran.nextDouble()*(half+x0);
		if(point <= (x0+half)/2.0) // Keep it
		    return point;
		else if(ran.nextDouble()*(0.2-0.02*(x0+half)) > 0.2-0.04*point)
		    return x0+half-point;
		else
		    return point;           
	    }
	}	

	///*
	// each from a guassian
	//return ran.nextGaussian()*0.5+x0;  
	//return ran.nextDouble()*range+x0-range/2;
	//*/
    }   
}

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
 *    MIDataPopulation.java
 *
 */
package milk.data;

import weka.core.Utils;
import java.lang.*;
import java.io.*;

/**
 * Generate n-dimensional 2-class MI data using instance-based approach
 * population version
 */
public class MIDataPopulation{   
    
    public static void main(String[] args) {  
	
	try{
	    MIDataPopulation data = new MIDataPopulation();
	    data.generateData(args);
	}catch (Exception ex) {
	    System.err.println(ex.getMessage());
	    ex.printStackTrace();
	}
    }
    
    public void generateData(String[] args) throws Exception{
	try{  
	    int index=0;
	    String fileName = args[index++];
	    int numEx = Integer.parseInt(args[index++]),
		numAtts = Integer.parseInt(args[index++]);
	    double[] beta = new double[numAtts];
	    for(int i=0; i<numAtts; i++)
		beta[i] = Double.parseDouble(args[index++]);
	    
	    double prob[]=new double[numEx], x[][]=new double[numEx][numAtts];
	    int cla[]=new int[numEx];
	    double wrong0=0, wrong1=0, wrong=0;
	    
	    for(int y=0; y < numEx; y++){
		for(int z=0; z<numAtts; z++){
		    // one centroid
		    x[y][z]=Math.random()*8-4; // A bag in [-4, 4) with range [x0-1, x0+1)
		}
		
		prob[y] = exemplarPosterior(x[y], beta);
		cla[y] = (Math.random() > prob[y])? 0:1;
		
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
	    pw.println("@relation "+fileName+"\n");
	    pw.print("@attribute examplar {");
	    
	    StringBuffer[] bagID = new StringBuffer[numEx]; 
	    for(int j=0; j<numEx; j++){
		bagID[j] = new StringBuffer(j+"_");
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
	    
	    for(int i=0; i<numAtts; i++)
		pw.println("@attribute t"+i+" numeric");
	    pw.println("@attribute class {0,1}");
	    pw.println("\n@data");
	    
	    for(int y=0; y < numEx; y++){               
		int size = (int)(Math.random()*5)+1;
		for(int a=0; a<size; a++){
		    pw.print(bagID[y].toString()+", ");
		    for(int b=0; b<numAtts; b++)
			pw.print(nextPoint(x[y][b])+", ");
		    pw.println(cla[y]);
		}
		
		System.out.print(y+" = (");
		for(int b=0; b<numAtts-1; b++)
		    System.out.print(x[y][b]+",");              
		System.out.println(x[y][numAtts-1]+"); p="+prob[y]+"; size="+size+": "+cla[y]);
	    }
	    pw.close();
	}catch(Exception e){
	    throw new Exception("MIData: java MI** <filename> <numExamplers> "+
				"<numAttributes> "+"<coefficient of Attribute1>"
				+" <coefficient of Attribute2> ...\n\n"+
				e.getMessage());
	}
    }
    
    // Generate random variates in a range of [x0-1, x0+1), with normalized
    // triangle distribution in [-5, 5): 0.2+0.04x if x<=0; 0.2-0.04x if x>0.
    protected double nextPoint(double x0) throws Exception{
	if(x0 >= 1){
	    double point = Math.random()*2.0+(x0-1);
	    if(point <= x0) // Keep it
		return point;
	    else if(Math.random()*(0.2-0.04*x0) > 0.2-0.04*point)
		return 2.0*x0-point;
	    else
		return point;       
	}
	else if(x0 <= -1){
	    double point = Math.random()*2.0+(x0-1);
	    if(point >= x0) // Keep it
		return point;
	    else if(Math.random()*(0.2+0.04*x0) > 0.2+0.04*point)
		return 2.0*x0-point;
	    else
		return point;    
	}
	else{
	    double total=-0.08*x0*x0+0.72, 
		left = -0.04*x0*x0-0.32*x0+0.36;
	    
	    if(Math.random()*total <= left){ // Left side
		double point = Math.random()*(1-x0)+x0-1;
		if(point >= (x0-1)/2.0) // Keep it
		    return point;
		else if(Math.random()*(0.2+0.02*(x0-1)) > 0.2+0.04*point)
		    return x0-1-point;
		else
		    return point;       
	    }
	    else{ // Right side
		double point = Math.random()*(1+x0);
		if(point <= (x0+1)/2.0) // Keep it
		    return point;
		else if(Math.random()*(0.2-0.02*(x0+1)) > 0.2-0.04*point)
		    return x0+1-point;
		else
		    return point;           
	    }
	}
    }
    
    // Compute the posterior probability of a bag from the centroid
    protected double exemplarPosterior(double[] x, double[] beta) 
	throws Exception{
	double logodds=0; // Prob. of this bag to be positive
	for(int z=0; z<x.length; z++){
	    // one centroid
	    double x0=x[z]; // A bag in [-5, 5) with range [x0-1, x0+1)     
	    if(x0>=1){                  
		double l = beta[z]*(-3*x0*x0+15*x0-1.0)/(-3*x0+15);
		logodds += l;
	    }
	    else if(x0<=-1){
		double l = beta[z]*(3*x0*x0+15*x0+1.0)/(3*x0+15);
		logodds += l;               
	    }
	    else{
		double l = beta[z]*(x0*x0*x0-12*x0)/(3*x0*x0-27);
		logodds += l;                   
	    }
	}
	
	return Math.exp(logodds)/(1.0+Math.exp(logodds));
    }
}

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
 *    TLDData.java
 *
 */
package milk.data;

import weka.core.RandomVariates;
import java.lang.*;
import java.io.*;
import weka.core.Utils;

/**
 * Generate n-dimensional 2-class MI data using two level distributions
 * prior = 0.5
 */
public class TLDData{   
    
    public static void main(String[] args) {		
	try{
	    int index=0;
	    String seedString = Utils.getOption('s', args);
	    long seed = 1;
	    if(seedString != "")
		seed = Long.parseLong(seedString);
	    String fileName = args[index++];
	    int numEx = Integer.parseInt(args[index++]),
		numAtts = Integer.parseInt(args[index++]);
	    
	    double[][] a = new double[2][numAtts], b = new double[2][numAtts], 
		w = new double[2][numAtts], m = new double[2][numAtts];
	    int SIZE = 0;
	    
	    for(int i=0; i<2; i++){
		for(int j=0; j<numAtts; j++){
		    a[i][j] = Double.parseDouble(args[index++]);
		    b[i][j] = Double.parseDouble(args[index++]);
		    w[i][j] = Double.parseDouble(args[index++]);
		    m[i][j] = Double.parseDouble(args[index++]);
		}
	    }	
	    RandomVariates ran = new RandomVariates(seed);
	    
	    PrintWriter pw = new PrintWriter(new FileWriter(fileName+".arff"));
	    pw.println("%Positive:");
	    for(int j=0; j<numAtts; j++)
		pw.println("%Att "+j+": a="+a[0][j]+", b="+b[0][j]+", w="+w[0][j]+", m="+m[0][j]);
	    pw.println("%Negative:");
	    for(int j=0; j<numAtts; j++)
		pw.println("%Att "+j+": a="+a[1][j]+", b="+b[1][j]+", w="+w[1][j]+", m="+m[1][j]);

	    double[][][] param = new double[numEx*2][numAtts][2]; // mu and sigma for each dimension and bag
	    
	    // class 0
	    int c=0;
	    for(int x=0; x<numEx; x++){	
		for(int y=0; y < numAtts; y++){			
		    double variance = ran.nextGamma(b[c][y]/2.0);
		    while(variance<=10e-9)
			variance = ran.nextGamma(b[c][y]/2.0);
		    double sigma = Math.sqrt(0.5*a[c][y]/variance),
			mu = ran.nextGaussian()*Math.sqrt(w[c][y])*sigma+m[c][y]; 
		    param[x][y][0] = mu;
		    param[x][y][1] = sigma;
		}
	    }
	    // class 1
	    c=1;
	    for(int x=numEx; x<2*numEx; x++){	
		for(int y=0; y < numAtts; y++){			
		    double variance = ran.nextGamma(b[c][y]/2.0);
		    while(variance<=10e-9)
			variance = ran.nextGamma(b[c][y]/2.0);
		    double sigma = Math.sqrt(0.5*a[c][y]/variance),
			mu = ran.nextGaussian()*Math.sqrt(w[c][y])*sigma+m[c][y]; 
		    param[x][y][0] = mu;
		    param[x][y][1] = sigma;
		}
	    }
	    
	    pw.println("%seed="+seed+"\n");
	    pw.println("@relation "+fileName+"\n");
	    pw.print("@attribute examplar {");

	    StringBuffer[] bagID = new StringBuffer[numEx*2]; 
	    for(int j=0; j<numEx*2; j++){
		bagID[j] = new StringBuffer(j+"_");		
		// centroid
		for(int k=0; k<numAtts-1; k++)
		    bagID[j].append(Utils.doubleToString(param[j][k][0],3,2)+"_"+
				    Utils.doubleToString(param[j][k][1],3,2)+"_");
		bagID[j].append(Utils.doubleToString(param[j][numAtts-1][0],3,2)+"_"+
				Utils.doubleToString(param[j][numAtts-1][1],3,2));
	    }
	    
	    for(int j=0; j<numEx*2; j++){
		pw.print(bagID[j].toString());
		if(j == (2*numEx-1))
		    pw.println("}");
		else
		    pw.print(",");
	    }
	    
	    for(int j=1; j<=numAtts; j++)
		pw.println("@attribute X"+j+" numeric");
	    pw.println("@attribute class {0,1}");
	    pw.println("\n@data");
	    
	    for(int y=0; y < 2*numEx; y++){
		int size = 1+ran.nextInt(20); //size		
		double[][] data = createEx(size, param[y], ran);

		for(int p=0; p<size; p++){
		    pw.print(bagID[y].toString()+", ");
		    for(int q=0; q<numAtts; q++){
			pw.print(data[p][q]+", ");
		    }
		    
		    if(y < numEx)
			pw.println(0);
		    else
			pw.println(1);
		}		
	    } 
	    	    
	    pw.close();
	}catch (Exception ex) {
	    System.err.println(ex.getMessage());
	    ex.printStackTrace();
	}
    }
    
    public static double[][] createEx(int num, double[][] param, RandomVariates ran){
	double[][] result = new double[num][param.length];
	
	for(int j=0; j<num; j++)
	    for(int k=0; k<param.length; k++)
		result[j][k] = param[k][0]+param[k][1]*ran.nextGaussian();
	
	return result;
    }
}




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
 *    BagStats.java
 *
 */
package milk.data;

import weka.core.*;
import milk.core.*;
import java.io.*;


public class BagStats {

    public static void main (String args[]) throws Exception {

	FileReader file = new FileReader(args[0]);
	Instances insts = new Instances(file);
	insts.setClassIndex(insts.numAttributes() - 1);

	Exemplars exs = new Exemplars(insts);

	double av = 0;
	int max = 0;
	int min = 123333333;
	double[] nums = new double[exs.numExemplars()];
	int pos=0, neg=0;

	for(int i=0; i<exs.numExemplars();i++) {
	    if(exs.exemplar(i).classValue() != 0)
		pos++;
	    else
		neg++;
	    
	    nums[i] = exs.exemplar(i).getInstances().numInstances();
	    av += exs.exemplar(i).getInstances().numInstances();
	    if (max < exs.exemplar(i).getInstances().numInstances()) max = exs.exemplar(i).getInstances().numInstances();
	    if (min > exs.exemplar(i).getInstances().numInstances()) min = exs.exemplar(i).getInstances().numInstances();
	}

	System.out.println("Number of bags: "+exs.numExemplars());
	System.out.println("Number of instances: "+insts.numInstances());
	System.out.println("Number of attributes (without id and class): "+(insts.numAttributes()-2));
	System.out.println("Average bag size: "+(av/exs.numExemplars()));
	System.out.println("Maximum bag size: "+max);
	System.out.println("Minimum bag size: "+min);
	System.out.println("Number of positive/negative bags: "+pos+"/"+neg);
	int[] order = Utils.sort(nums);
	double median;
	int half = exs.numExemplars()/2;
	if(half*2 == exs.numExemplars()){
	    System.out.println("even: "+half);	    
	    median = (nums[order[half-1]]+nums[order[half]])/2.0;
	}
	else{
	    System.out.println("odd: "+half);
	    median = nums[order[half]]/2.0;
	}
	
	System.out.println("Median bag size: "+median);
    }
}

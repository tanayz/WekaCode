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
 *    MIRBFNetwork.java
 *    Copyright (C) 2004 Eibe Frank
 *
 */
package milk.classifiers;

import milk.core.Exemplars;
import milk.core.Exemplar;
import weka.core.Attribute;
import weka.core.Option;
import weka.core.Utils;
import java.util.Vector;
import java.util.Enumeration;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ClusterMembership;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.OptionHandler;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.EM;
import weka.clusterers.MakeDensityBasedClusterer;

/**
 * 
 * Multi-instance RBF network. Uses k-means with distributions fit in post-processing step
 * plus MILR at the second level.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.13 $ 
 */
public class MIRBFNetwork extends MIClassifier implements OptionHandler {  

  /** The logistic regression model */
  protected MIClassifier m_logistic = new MILR();

  /** The RBF filter */
  protected ClusterMembership m_clm = new ClusterMembership();

  /** The number of clusters to use */
  protected int m_num_clusters = 10;

  /** The ridge regression coefficient for logistic regression */
  protected double m_ridge = 1e-6;

  /**
   * Get the Num_clusters value.
   * @return the Num_clusters value.
   */
  public int getNumClusters() {
    return m_num_clusters;
  }

  /**
   * Set the Num_clusters value.
   * @param newNum_clusters The new Num_clusters value.
   */
  public void setNumClusters(int newNum_clusters) {
    this.m_num_clusters = newNum_clusters;
  }

  /**
   * Get the Ridge value.
   * @return the Ridge value.
   */
  public double getRidge() {
    return m_ridge;
  }

  /**
   * Set the Ridge value.
   * @param newRidge The new Ridge value.
   */
  public void setRidge(double newRidge) {
    this.m_ridge = newRidge;
  }
    
  /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector(2);
    newVector.addElement(new Option("\tThe number of clusters to use.",
				    "N", 1, "-N"));
    newVector.addElement(new Option("\tSet the ridge in the log-likelihood.",
				    "R", 1, "-R <ridge>"));
    return newVector.elements();
  }
    
  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -N number <br>
   * The number of clusters to use.<p>
   *
   * -R ridge <br>
   * Set the ridge parameter for the log-likelihood.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    String ridgeString = Utils.getOption('R', options);
    if (ridgeString.length() != 0) {
      m_ridge = Double.parseDouble(ridgeString);
    } else {
      m_ridge = 1.0e-6;
    }
    
    String numString = Utils.getOption('N', options);
    if (numString.length() != 0) {
      m_num_clusters = Integer.parseInt(numString);
    } else {
      m_num_clusters = 10;
    }
  }
    
  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {
    
    String [] options = new String [4];
    int current = 0;
    
    options[current++] = "-R";
    options[current++] = ""+m_ridge;
    options[current++] = "-N";
    options[current++] = ""+m_num_clusters;
    
    while (current < options.length) 
      options[current++] = "";
    return options;
  }
  
  // Implements transformation for training data
  public Exemplars transform(Exemplars ex) throws Exception {

    // Throw all the instances together
    Instances data = new Instances(ex.exemplar(0).getInstances());
    for (int i = 0; i < ex.numExemplars(); i++) {
      Exemplar curr = ex.exemplar(i);
      double weight = 1.0 / (double)curr.getInstances().numInstances();
      for (int j = 0; j < curr.getInstances().numInstances(); j++) {
	Instance inst = (Instance)curr.getInstances().instance(j).copy();
	inst.setWeight(weight);
	data.add(inst);
      }
    }
    double factor = (double) data.numInstances() / (double) data.sumOfWeights(); 
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(data.instance(i).weight() * factor);
    }

    SimpleKMeans kMeans = new SimpleKMeans();
    kMeans.setNumClusters(m_num_clusters);
    MakeDensityBasedClusterer clust = new MakeDensityBasedClusterer();
    clust.setClusterer(kMeans);
    m_clm.setDensityBasedClusterer(clust);
    m_clm.setIgnoredAttributeIndices("" + (ex.exemplar(0).idIndex() + 1));
    m_clm.setInputFormat(data);

    // Use filter and discard result
    Instances tempData = Filter.useFilter(data, m_clm);
    tempData = new Instances(tempData, 0);
    tempData.insertAttributeAt(ex.exemplar(0).getInstances().attribute(0), 0);

    // Go through exemplars and add them to new dataset
    Exemplars newExs = new Exemplars(tempData);
    for (int i = 0; i < ex.numExemplars(); i++) {
      Exemplar curr = ex.exemplar(i);
      Instances temp = Filter.useFilter(curr.getInstances(), m_clm);
      temp.insertAttributeAt(ex.exemplar(0).getInstances().attribute(0), 0);
      for (int j  = 0; j < temp.numInstances(); j++) {
	temp.instance(j).setValue(0, curr.idValue());
      }
      newExs.add(new Exemplar(temp));
    }
    //System.err.println("Finished transforming");
    //System.err.println(newExs);
    return newExs;
  }

  // Implements transformation for test instance
  public Exemplar transform(Exemplar test) throws Exception{

    Instances temp = Filter.useFilter(test.getInstances(), m_clm);
    temp.insertAttributeAt(test.getInstances().attribute(0), 0);
    for (int j  = 0; j < temp.numInstances(); j++) {
      temp.instance(j).setValue(0, test.idValue());
      //System.err.println(temp.instance(j));
    }
    return new Exemplar(temp);
  }
    
  /**
   * Builds the classifier.
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

    ((MILR)m_logistic).setRidge(m_ridge);
    m_logistic.buildClassifier(transform(train));
  }		
    
  /**
   * Computes the distribution for a given exemplar
   *
   * @param exmp the exemplar for which distribution is computed
   * @return the distribution
   * @exception Exception if the distribution can't be computed successfully
   */
  public double[] distributionForExemplar(Exemplar exmp) throws Exception {
	
    return m_logistic.distributionForExemplar(transform(exmp));
  }
       
  /**
   * Gets a string describing the classifier.
   *
   * @return a string describing the classifer built.
   */
  public String toString() {	
    return "MIRBFNetwork: \n\n" + 
      m_logistic.toString();
  }
    
    /**
     * Main method for testing this class.
     *
     * @param argv should contain the command line arguments to the
     * scheme (see Evaluation)
     */
    public static void main(String [] argv) {
	try {
	    System.out.println(MIEvaluation.evaluateModel(new MIRBFNetwork(), argv));
	} catch (Exception e) {
	    e.printStackTrace();
	  System.err.println(e.getMessage());
	}
    }
}

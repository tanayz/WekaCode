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
 *    Exemplar.java
 *    Copyright (C) 2002 University of Waikato
 * 
 */

package milk.core;

import java.io.*;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.*;

/**
 * Class for handling an ordered set of weighted exemplars. <p>
 *
 * This class is not intended for reading directly from the Reader.
 * Instead, the data should be pre-processed and then are tried to
 * form an exemplar.
 * 
 * The following members are still useful from the Instances class
 * in the context of exemplar by using getInstances() 
 *
 * <br> attribute(int) 
 * <br> attribute(String) 
 * <br> attributeStats(int) 
 * <br> attributeToDoubleArray(int) 
 * <br> checkForStringAttributes() 
 * <br> checkInstance(Instance) 
 * <br> classAttribute() 
 * <br> classIndex() 
 * <br> compactify() 
 * <br> delete() 
 * <br> delete(int) 
 * <br> deleteWithMissing(Attribute) 
 * <br> deleteWithMissing(int)  
 * <br> enumerateInstances() 
 * <br> firstInstance() 
 * <br> instance(int) 
 * <br> lastInstance() 
 * <br> meanOrMode(Attribute) 
 * <br> meanOrMode(int) 
 * <br> numAttributes() 
 * <br> numClasses() 
 * <br> numDistinctValues(Attribute) 
 * <br> numDistinctValues(int) 
 * <br> numInstances() 
 * <br> randomize(Random) 
 * <br> renameAttribute(Attribute, String) 
 * <br> renameAttribute(int, String) 
 * <br> renameAttributeValue(Attribute, String, String) 
 * <br> renameAttributeValue(int, int, String) 
 * <br> resample(Random) 
 * <br> resampleWithWeights(Random) 
 * <br> resampleWithWeights(Random, double[]) 
 * <br> sort(Attribute) 
 * <br> sort(int) 
 * <br> stringFreeStructure()  
 * <br> sumOfWeights() 
 * <br> variance(Attribute) 
 * <br> variance(int) 
 *
 * Typical usage is to read the instances from the Reader and try to feed
 * it into into an array of Exemplars according to it's ID values 
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.33 $ 
 */
public class Exemplar implements Serializable {
 
    /** The exemplar's ID attribute */
    private int m_IdIndex;

    /** The value of the ID of this exemplar */
    private double m_IdValue;

    /** The class index of this exemplar */
    private int m_ClassIndex;
    
    /** The class value of this exemplar */
    private double m_ClassValue;

    /** The instances in the exemplar*/
    private Instances m_Instances;

    /** The weight of this exemplar */
    private double m_Weight = 1;

    /**
     * Constructor using one instance to form an exemplar
     * 
     * @param instance the given instance
     * @param id the ID index
     */
    public Exemplar(Instance inst, int id) {
	m_IdIndex = id;
	m_IdValue = inst.value(id);
	m_ClassIndex = inst.classIndex();
	m_ClassValue = inst.classValue();
	m_Instances = new Instances(inst.dataset(),1);
	m_Instances.add(inst);
    }
    
    /** 
     * Constructor creating an empty Exemplar with the same structure
     * of the given exemplar and the given size
     *
     * @param exemplar the given exemplar
     * @param size the given size
     */
    public Exemplar(Exemplar exemplar, int size){
	m_IdIndex = exemplar.m_IdIndex;
	m_IdValue = exemplar.m_IdValue;
	m_ClassIndex = exemplar.m_ClassIndex;
	m_ClassValue = exemplar.m_ClassValue;
	m_Instances = new Instances(exemplar.getInstances(), size);
    }

    /**
     * Constructor to form an exemplar by copying from another exemplar
     *
     * @param exemplar the copied exemplar
     */
    public Exemplar(Exemplar exemplar){
	m_IdIndex = exemplar.m_IdIndex;
	m_IdValue = exemplar.m_IdValue;
	m_ClassIndex = exemplar.m_ClassIndex;
	m_ClassValue = exemplar.m_ClassValue;
	m_Instances = new Instances(exemplar.m_Instances);
	m_Weight = exemplar.m_Weight;
    }

    /**
     * Constructor copying all instances and references to
     * the header information from the given set of instances.
     * Set the attribute with first index as exemplar ID
     * 
     * @param instances the set to be copied
     * @exception if the exemplar cannot be built properly
     */
    public Exemplar(Instances dataset) throws Exception {
	this(dataset, 0);
    }
    
    /**
     * Constructor creating an exemplar with the given dataset and the 
     * given ID index
     *
     * @param instances the instances from which the header 
     * information is to be taken
     * @param id the index of the ID of the exemplar 
     */
    public Exemplar(Instances dataset, int id) throws Exception{
	m_IdIndex = id;
	m_ClassIndex = dataset.classIndex();
	m_Instances = new Instances(dataset);
    
	if(!m_Instances.attribute(m_IdIndex).isNominal())
	    throw new Exception("The exempler's ID is not nominal!");
	
	double idvalue = (m_Instances.firstInstance()).value(m_IdIndex);
	double clsvalue = (m_Instances.firstInstance()).classValue();
	
	// The the validity of this exemplar
	for(int i=1; i < m_Instances.numInstances(); i++){
	    Instance inst = m_Instances.instance(i);
	    if((!Utils.eq(inst.value(m_IdIndex), idvalue)) 
	       //|| (!Utils.eq(inst.classValue(), clsvalue))
	       )
		throw new Exception("The Id value and/or class value is not unique!");
	}
	
	m_IdValue = idvalue;
	m_ClassValue = clsvalue;
    }
    
    /**
     * Adds one instance to the end of the set. 
     * Shallow copies instance before it is added. Increases the
     * size of the dataset if it is not large enough. Does not
     * check if the instance is compatible with the dataset.
     *
     * @param instance the instance to be added
     */
    public final void add(Instance instance) {	
	Instance inst = (Instance)instance.copy();
	
	if(!checkInstance(instance))
	    throw new IllegalArgumentException("The Id value and/or class value " +
					       "is not compatible: add failed.");
	else
	    m_Instances.add(inst);
    }

    /**
     * Checks if the given instance is compatible with this 
     * Exemplar. 
     *
     * @return true if the instance is compatible with the exemplar 
     */
    public final boolean checkInstance(Instance instance){
	if(!m_Instances.checkInstance(instance))
	    return false;
	
	if((!Utils.eq(instance.value(m_IdIndex), m_IdValue)) 
	   //|| (!Utils.eq(instance.classValue(), m_ClassValue))
	   )
	    return false;
	
	return true;
    }

    /**
     * Returns the class attribute.
     *
     * @return the class attribute
     * @exception UnassignedClassException if the class is not set
     */
    public final Attribute classAttribute() {
	
	if (m_ClassIndex < 0) {
	    throw new UnassignedClassException("Class index is negative (not set)!");
	}
	return m_Instances.attribute(m_ClassIndex);
    }
    
    /**
     * Returns the class attribute's index. Returns negative number
     * if it's undefined.
     *
     * @return the class index as an integer
     */
    public final int classIndex() {
	
	return m_ClassIndex;
    }
    
    /**
     * Returns the class value of this exemplar.
     *
     * @return the class value
     * @exception UnassignedClassException if the class is not set
     */
    public final double classValue(){
	if (m_ClassIndex < 0) {
	    throw new UnassignedClassException("Class index is negative (not set)!");
	}
	return m_ClassValue;
    }    
    
    /**
     * Sets the class value of the exemplar
     *
     *@param cv the new class value
     */
    public void setClassValue(double cv) {
        m_ClassValue = cv;
    }
    
    /**
     * Compactifies the set of instances in this exemplar
     */
    public final void compactify() {
	
	m_Instances.compactify();
    }

    /**
     * Deletes an attribute at the given position 
     * (0 to numAttributes() - 1). A deep copy of the attribute
     * information is performed before the attribute is deleted.
     *
     * @param pos the attribute's position
     * @exception IllegalArgumentException if the given index is out of range or the
     * class attribute is being deleted
     */
    public void deleteAttributeAt(int position) throws IllegalArgumentException{
	if (m_ClassIndex > position)
	    m_ClassIndex--;
	if (m_IdIndex > position)
	    m_IdIndex--;
	
	m_Instances.deleteAttributeAt(position);
    }
    

    /**
     * Returns an enumeration of all the attributes.
     *
     * @return enumeration of all the attributes.
     */
    public Enumeration enumerateAttributes() {
	FastVector vector = new FastVector(0);
	for(int i=0; i < m_Instances.numAttributes(); i++)
	    if((i != m_ClassIndex) && (i != m_IdIndex))
		vector.addElement((Object)m_Instances.attribute(i));
       
	return vector.elements();
    }    

    /**
     * Returns the dataset in this exemplar
     *
     * @return all the instances in the exemplar
     */
    public Instances getInstances(){
	return m_Instances;
    }
    
   /**
     * Returns the ID attribute.
     *
     * @return the ID attribute
     */
    public final Attribute idAttribute() {
	return m_Instances.attribute(m_IdIndex);
    }
    
    /**
     * Returns the ID attribute's index. 
     *
     * @return the ID index as an integer
     */
    public final int idIndex() {	
	return m_IdIndex;
    }

    /**
     * Returns the ID attribute's value. 
     *
     * @return the ID value of this exemplar
     */
    public final double idValue() {	
	return m_IdValue;
    }
    
    /**
     * Inserts an attribute at the given position (0 to 
     * numAttributes()) and sets all values to be missing.
     * Shallow copies the attribute before it is inserted, and performs
     * a deep copy of the existing attribute information.
     *
     * @param att the attribute to be inserted
     * @param pos the attribute's position
     * @exception IllegalArgumentException if the given index is out of range
     */
    public void insertAttributeAt(Attribute att, int position) {
	
	if (m_ClassIndex >= position) 
	    m_ClassIndex++;
	if (m_IdIndex >= position) 
	    m_IdIndex++;
	m_Instances.insertAttributeAt(att, position);
    }

    /**
     * Returns the mean (mode) for all attributes (except ID and class) as
     * a floating-point value. 
     * Returns 0 if the attribute is neither nominal nor 
     * numeric. If all values are missing it returns zero.
     *
     * @return the mean or the mode of all attributes
     */
    public final double[] meanOrMode() {
	int numAttr = m_Instances.numAttributes()-2; // Excluding ID and class
	double[] mean = new double[numAttr];
	int j=0;

	for(int i=0; i < (numAttr+2); i++){	    
	    if((i != m_IdIndex) && (i != m_ClassIndex)){
		if(Utils.gr(m_Instances.sumOfWeights(),0.0) && 
		   !isAllMissing(i))
		    mean[j] = m_Instances.meanOrMode(i);
		else
		    mean[j] = Double.NaN;
		j++;
	    }
	}
	 
	return mean;
    }

    /**
     * Computes the variances for all numeric attributes.
     * For nominal attribute, the variance is set to -1
     *
     * @return the variances for all numeric attributes
     */
    public final double[] variance() {
	int numAttr = m_Instances.numAttributes()-2; // Excluding ID and class
	double[] var = new double[numAttr];
	int j=0;
	for(int i=0; i < (numAttr+2); i++){
	    if((i != m_IdIndex) && (i != m_ClassIndex)){
		if(m_Instances.attribute(i).isNumeric())
		    var[j] = m_Instances.variance(i);
		else
		    var[j] = -1;
		
		j++;
	    }
	}
	
	return var;
    }
    
    /**
     * Returns the number of ID labels.
     *
     * @return the number of ID labels.
     */
    public final int numIds() {
	return idAttribute().numValues();
    }
    
    /**
     * Sets the weight of the exemplar.
     *
     * @param weight the weight
     */
    public final void setWeight(double weight) {
	m_Weight = weight;
    }
    
    /**
     * Returns the exemplar as a string. Strings
     * are quoted if they contain whitespace characters, or if they
     * are a question mark.
     *
     * @return the exemplar as a string
     */
    public final String toString() {
	
	StringBuffer text = new StringBuffer();
	
	Attribute id = idAttribute();
	Attribute cl = classAttribute();
	text.append("@Exemplar: \nID: " + id.name()+ " = " + 
		    id.value((int)m_IdValue) + "\nClass: " + 
		    cl.name() + " = " + cl.value((int)m_ClassValue) +"\n\n");
	
	for (int i = 0; i < m_Instances.numAttributes(); i++) {
	    text.append(m_Instances.attribute(i));
	    if(i==m_IdIndex)
		text.append(" (ID Attribute)");
	    else if(i==m_ClassIndex)
		text.append(" (Class Attribute)");

	    text.append("\n");
	}
	
	text.append("\n@data\n");
	for (int i = 0; i < m_Instances.numInstances(); i++) {
	    text.append(m_Instances.instance(i)+"\n");
	    }
	return text.toString();
    }  
    
    /**
     * Returns the exemplar's weight.
     *
     * @return the exemplar's weight as a double
     */
    public final double weight() {
	return m_Weight;
    }

    /**
     * Check whether the values of the specified attribute in this exemplar
     * are all missing values
     *
     * @param attIndex the specified attribute index
     * @return whether values are all missing
     */ 
    public final boolean isAllMissing(int attIndex){
	for(int i=0; i<m_Instances.numInstances(); i++)
	    if(!m_Instances.instance(i).isMissing(attIndex))
		return false;
	return true;
    }
    
  /**
   * Main method for testing this class -- just prints out a set
   * of Exemplars.  Assume the ID index is 0.
   *
   * @param argv should contain one element: the name of an ARFF file
   */
  public static void main(String [] args) {

    try {
      Reader r = null;
      if (args.length > 1) {
	throw (new Exception("Usage: Instances <filename>"));
      } else if (args.length == 0) {
        r = new BufferedReader(new InputStreamReader(System.in));
      } else {
        r = new BufferedReader(new FileReader(args[0]));
      }
      Instances i = new Instances(r);
      i.setClassIndex(i.numAttributes()-1);

      Attribute id = i.attribute(0);
      if(!id.isNominal())
	  throw new Exception("The first attribute is not nominal");

      Exemplar[] egs = new Exemplar[id.numValues()];
      for(int j=0; j < egs.length; j++)
	  egs[j] = null;

      for(int j=0; j < i.numInstances(); j++){
	  Instance ins = i.instance(j);
	  int idv = (int)ins.value(0);
	  if(egs[idv] == null)
	      egs[idv] = new Exemplar(ins, 0);
	  else
	      egs[idv].add(ins);
      }

      for(int j=0; j < egs.length; j++)
	  System.out.println(egs[j].toString());
    } catch (Exception ex) {
      System.err.println(ex.getMessage());
    }
  }
}

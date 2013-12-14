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
 *    Exemplars.java
 *    Copyright (C) 2002 University of Waikato
 *
 */
package milk.core;

import java.io.*;
import java.util.*;
import weka.core.*;

/**
 * The class of a set of exemplars
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.1 $ 
 */
public class Exemplars implements Serializable {
 
    /** The dataset's name. */
    private String m_RelationName;         
    
    /** The attribute information. */
    private Attribute[] m_Attributes;
    
    /** The exemplars. */
    private Vector m_Exemplars;
    
    /** The exemplars' ID attribute */
    private int m_IdIndex;
    
    /** The class index of this exemplar */
    private int m_ClassIndex;   
    
    
    /** 
     * Constructor to form an Exemplars by deep copying from 
     * another Exemplars
     *
     * @param exemplars the copied Exemplars
     */
    public Exemplars(Exemplars exemplars){
	m_IdIndex = exemplars.m_IdIndex;
	m_ClassIndex = exemplars.m_ClassIndex;
	m_RelationName = exemplars.m_RelationName;
	
	int attLen = exemplars.m_Attributes.length;
	m_Attributes = new Attribute[attLen];
	for(int i=0; i < attLen; i++)
	    m_Attributes[i] = (Attribute)exemplars.m_Attributes[i].copy();
	
	int exLen = exemplars.numExemplars();
	m_Exemplars = new Vector(exLen);
	for(int i=0; i < exLen; i++)
	    m_Exemplars.addElement(new Exemplar(exemplars.exemplar(i)));
    }

    /** 
     * Constructor creating an empty Exemplars with the same structure
     * of the given Exemplars and the given size (i.e. the number of
     * exemplars in the set)
     *
     * @param exemplars the given Exemplars
     * @param size the given size
     */
    public Exemplars(Exemplars exemplars, int size){
	m_IdIndex = exemplars.m_IdIndex;
	m_ClassIndex = exemplars.m_ClassIndex;
	m_RelationName = exemplars.m_RelationName;
	
	int attLen = exemplars.m_Attributes.length;
	m_Attributes = new Attribute[attLen];
	for(int i=0; i < attLen; i++)
	    m_Attributes[i] = (Attribute)exemplars.m_Attributes[i].copy();
	
	m_Exemplars = new Vector(size);
    }

    /**
     * Constructor using the given dataset and set ID index to 0
     * 
     * @param dataset the set to be copied
     * @exception Exception if the class index of the dataset 
     * is not set(i.e. -1)
     */
    public Exemplars(Instances dataset) throws Exception{
	
	this(dataset, 0);
    }

  /**
   * Creates a new set of instances by copying a 
   * subset of another set.
   *
   * @param source the set of instances from which a subset 
   * is to be created
   * @param first the index of the first instance to be copied
   * @param toCopy the number of instances to be copied
   * @exception IllegalArgumentException if first and toCopy are out of range
   */
  public Exemplars(Exemplars source, int first, int toCopy) {
    
    this(source, toCopy);

    if ((first < 0) || ((first + toCopy) > source.numExemplars())) {
      throw new IllegalArgumentException("Parameters first and/or toCopy out "+
                                         "of range");
    }
    source.copyExemplars(first, this, toCopy);
  }

  /**
   * Constructor using the given dataset and set ID index to 
   * the given ID index.  Any instances with class value or ID
   * value missing will be dropped.
   *
   * @param dataset the instances from which the header 
   * information is to be taken
   * @param idIndex the ID attribute's index 
   * @exception Exception if the class index of the dataset 
   * is not set(i.e. -1) or the data is not a multi-instance data
   */
    public Exemplars(Instances dataset, int idIndex) throws Exception {
	if(dataset.classIndex() == -1)
	    throw new Exception(" Class Index negative (class not set yet)!");
	
	m_ClassIndex =dataset.classIndex();
	m_RelationName= dataset.relationName();
	int numAttr = dataset.numAttributes();         
	m_Attributes = new Attribute[numAttr];
	for(int i=0; i < numAttr; i++)
	    m_Attributes[i] = dataset.attribute(i);

	m_IdIndex = idIndex;
	Attribute id = m_Attributes[m_IdIndex];
	if((m_IdIndex > numAttr) || (m_IdIndex < 0) 
	   || (!id.isNominal()))
	    throw new Exception ("ID index is wrong!");
	
	
	m_Exemplars = new Vector(id.numValues());
	
	for(int j=0; j < dataset.numInstances(); j++){
	    Instance ins = dataset.instance(j);
	    add(ins);
	}
    }

    /**
     * Adds one instance to one of the exemplars 
     *
     * @param instance the instance to be added
     * @exception Exception if the instance cannot be added properly
     */
    public final void add(Instance instance) {	
	Instance ins = (Instance)instance.copy();
	
	int idv = (int)ins.value(m_IdIndex);
	int x=0;
	for(; x < m_Exemplars.size(); x++){
	    Exemplar ex = (Exemplar)m_Exemplars.elementAt(x);
	    if(ex != null){
		if((int)(ex.idValue()) == idv){
		    if(!ex.checkInstance(instance))
			throw new IllegalArgumentException("Instance not compatible " +
							   "with the data");
		    ex.add(ins);
		    break;
		}
	    }
	}
	if(x == m_Exemplars.size()){
	    Exemplar ex = new Exemplar(ins, m_IdIndex);
	    ex.setWeight(1.0);
	    m_Exemplars.addElement(ex);
	}
    }


    /**
     * Adds one exemplar to the exemplars 
     *
     * @param exemplar the exemplar to be added
     * @exception Exception if the exemplar already exists
     */
    public final void add(Exemplar exemplar) {
	int idv = (int)exemplar.idValue();	
	for(int x=0; x < m_Exemplars.size(); x++){
	    Exemplar ex = (Exemplar)m_Exemplars.elementAt(x);
	    if((int)(ex.idValue()) == idv)
		throw new 
		  IllegalArgumentException("Exemplar already exists in the Exemplars");
	}	
	m_Exemplars.addElement(new Exemplar(exemplar));
    }
	
    /**
     * Returns an attribute.
     *
     * @param index the attribute's index
     * @return the attribute at the given position
     */ 
    public final Attribute attribute(int index) {
	
	return m_Attributes[index];
    }
    
    /**
     * Returns an attribute given its name. If there is more than
     * one attribute with the same name, it returns the first one.
     * Returns null if the attribute can't be found.
     *
     * @param name the attribute's name
     * @return the attribute with the given name, null if the
     * attribute can't be found
     */ 
    public final Attribute attribute(String name) {
	
	for (int i = 0; i < numAttributes(); i++) {
	    if (attribute(i).name().equals(name)) {
		return attribute(i);
	    }
	}
	return null;
    }

    /**
     * Checks for string attributes in the Exemplars
     *
     * @return true if string attributes are present, false otherwise
     */
    public boolean checkForStringAttributes() {
	
	int i = 0;
	
	while (i < m_Attributes.length) {
	    if (attribute(i++).isString()) {
		return true;
	    }
	}
	return false;
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
	return attribute(m_ClassIndex);
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
     * Compactifies each exemplar in this Exemplars
     */
    public final void compactify() {
	
	for(int i=0; i < m_Exemplars.size(); i++)
	    ((Exemplar)m_Exemplars.elementAt(i)).compactify();
	m_Exemplars.trimToSize();
    }

  /**
   * Copies instances from one set to the end of another 
   * one.
   *
   * @param source the source of the instances
   * @param from the position of the first instance to be copied
   * @param dest the destination for the instances
   * @param num the number of instances to be copied
   */
  private void copyExemplars(int from, Exemplars dest, int num) {
    
    for (int i = 0; i < num; i++) {
      dest.add(exemplar(from + i));
    }
  }
      
    /**
     * Removes all Exemplars from the set.
     */
    public final void delete() {
	m_Exemplars.removeAllElements();
    }
    
    /**
     * Removes an exemplar at the given position from the set.
     *
     * @param index the instance's position
     */
    public final void delete(int index) {
	m_Exemplars.removeElementAt(index);
	
    }
    
    /**
     * Deletes an attribute at the given position 
     * (0 to numAttributes() - 1). 
     *
     * @param pos the attribute's position
     * @exception Exception if the given index is out of range or the
     * class attribute is being deleted
     */
    public void deleteAttributeAt(int position) throws Exception{
	for(int i=0; i < m_Exemplars.size(); i++)
	    ((Exemplar)m_Exemplars.elementAt(i)).deleteAttributeAt(position);
	
	Exemplar eg = (Exemplar)m_Exemplars.firstElement();
	int len = m_Attributes.length -1;
	for(int j=position; j < len; j++)
	    m_Attributes[j] = eg.getInstances().attribute(j);
	m_Attributes[len] = null;

	m_ClassIndex = eg.classIndex();
	m_IdIndex = eg.idIndex();
    }

    /**
     * Deletes all string attributes in the dataset. A deep copy of the attribute
     * information is performed before an attribute is deleted.
     *
     * @exception IllegalArgumentException if string attribute couldn't be 
     * successfully deleted (probably because it is the class attribute).
     */
    public void deleteStringAttributes() throws Exception{
	
	int i = 0;
	while (i < m_Attributes.length) {
	    if (attribute(i).isString()) {
		deleteAttributeAt(i);
	    } else {
		i++;
	    }
	}
    }
    
    /**
     * Removes all instances with missing values for a particular
     * attribute from the dataset.
     *
     * @param attIndex the attribute's index
     */
    public final void deleteWithMissing(int attIndex) {
	
	for(int i=0; i < m_Exemplars.size(); i++)
	    ((Exemplar)m_Exemplars.elementAt(i)).getInstances().deleteWithMissing(attIndex);
    }
    
    /**
     * Removes all instances with missing values for a particular
     * attribute from the dataset.
     *
     * @param att the attribute
     */
    public final void deleteWithMissing(Attribute att) {
	
	deleteWithMissing(att.index());
    }
   
    /**
     * Returns an enumeration of all the attributes.
     *
     * @return enumeration of all the attributes.
     */
    public Enumeration enumerateAttributes() {
	
	return ((Exemplar)m_Exemplars.firstElement()).enumerateAttributes();
    }   
    
    /**
     * Returns a vector of exemplars in this Exemplars.
     *
     * @return a vector of all exemplars
     */
    public final Vector getExemplars() {
	
	return m_Exemplars;
    }
    
    /**
     * Returns the first exemplar in the set.
     *
     * @return the first exemplar in the set
     */
    public final Exemplar firstExemplar() {
	
	return (Exemplar)m_Exemplars.firstElement();
    }
    
    /**
     * Returns the ID attribute.
     *
     * @return the ID attribute
     */
    public final Attribute idAttribute() {
	
	return attribute(m_IdIndex);
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
     * Inserts an attribute at the given position (0 to 
     * numAttributes()) and sets all values to be missing.
     *
     * @param att the attribute to be inserted
     * @param pos the attribute's position
     * @exception IllegalArgumentException if the given index is out of range
     */
    public void insertAttributeAt(Attribute att, int position) {
	for(int i=0; i < m_Exemplars.size(); i++)
	    ((Exemplar)m_Exemplars.elementAt(i)).insertAttributeAt(att, position);
	
	Exemplar eg = (Exemplar)m_Exemplars.firstElement();
	int len = m_Attributes.length + 1;
	m_Attributes = new Attribute[len];
	for(int j=position; j < len; j++)
	    m_Attributes[j] = eg.getInstances().attribute(j);
	
	m_ClassIndex = eg.classIndex();
	m_IdIndex = eg.idIndex();	
    }
    
    /**
   * Returns the exemplar at the given position.
   *
   * @param index the exemplar's index
   * @return the exemplar at the given position
   */
    public final Exemplar exemplar(int index) {
	
	return (Exemplar)m_Exemplars.elementAt(index);
    }
    
    /**
     * Returns the last exemplar in the set.
     *
     * @return the last exemplar in the set
     */
    public final Exemplar lastExemplar() {
	return (Exemplar)m_Exemplars.lastElement();
    }
    
    /**
     * Returns the number of attributes.
     *
     * @return the number of attributes as an integer
     */
    public final int numAttributes() {
	
	return m_Attributes.length;
    }
    
    /**
     * Returns the number of class labels.
     *
     * @return the number of class labels as an integer if the class 
     * attribute is nominal, 1 otherwise.
     * @exception UnassignedClassException if the class is not set
     */
    public final int numClasses() {
	
	if (m_ClassIndex < 0) {
	    throw new UnassignedClassException("Class index is negative (not set)!");
	}
	if (!classAttribute().isNominal()) {
	    return 1;
	} else {
	    return classAttribute().numValues();
	}
    }
 
    /**
     * Returns the number of exemplars in the set.
     *
     * @param att the attribute
     * @return the number of distinct values of exemplar
     */
    public final int numExemplars() {
	
	return m_Exemplars.size();
    }
    
    /**
     * Returns the number of instances in the dataset.
     *
     * @return the number of instances in the dataset as an integer array
     */
    public final int[] numsInstances() {

	int[] size = new int[numExemplars()];
	for(int i=0; i < m_Exemplars.size(); i++)
	   size[i] = ((Exemplar)m_Exemplars.elementAt(i)).getInstances().numInstances();
	
	return size;
    }
    
    /**
     * Shuffles the exemplars in the set so that they are ordered 
     * randomly.
     *
     * @param random a random number generator
     */
    public final void randomize(Random random) {
	
	for (int j = numExemplars() - 1; j > 0; j--)
	    swap(j, random.nextInt(j+1));
    }
    
    /**
     * Returns the relation's name.
     *
     * @return the relation's name as a string
     */
    public final String relationName() {
	
	return m_RelationName;
    }
    
    /**
     * Renames an attribute.
     *
     * @param att the attribute's index
     * @param name the new name
     */
    public final void renameAttribute(int att, String name) {

	for(int i=0; i < m_Exemplars.size(); i++)
	    ((Exemplar)m_Exemplars.elementAt(i)).getInstances().renameAttribute(att, name);	
    }
    
    /**
     * Renames an attribute.
     *
     * @param att the attribute
     * @param name the new name
     */
    public final void renameAttribute(Attribute att, String name) {

	renameAttribute(att.index(), name);
    }
    
    /**
     * Renames the value of a nominal (or string) attribute value.
     *
     * @param att the attribute's index
     * @param val the value's index
     * @param name the new name 
     */
    public final void renameAttributeValue(int att, int val, String name) {
	
	for(int i=0; i < m_Exemplars.size(); i++)
	    ((Exemplar)m_Exemplars.elementAt(i)).getInstances().renameAttributeValue(att, val, name);	
    }
    
    /**
     * Renames the value of a nominal (or string) attribute value.
     *
     * @param att the attribute
     * @param val the value
     * @param name the new name
     */
    public final void renameAttributeValue(Attribute att, String val, 
					   String name) {
	
	int v = att.indexOfValue(val);
	if (v == -1) throw new IllegalArgumentException(val + " not found");
	renameAttributeValue(att.index(), v, name);
    }
    
    /**
     * Creates a new Exemplars of the same size using random sampling
     * with replacement.
     *
     * @param random a random number generator
     * @return the new Exemplars
     */
    public final Exemplars resample(Random random) {
	
	Exemplars newData = new Exemplars(this, numExemplars());
	while(newData.m_Exemplars.size() < numExemplars()){
	    int j = (int) (random.nextDouble() * (double) numExemplars());
	    newData.m_Exemplars.addElement(new Exemplar(exemplar(j)));
	}
	return newData;
    }
    
    /**
     * Creates a new Exemplars of the same size using random sampling
     * with replacement according to the current exemplar weights. The
     * weights of the exemplars in the new set are set to one.
     *
     * @param random a random number generator
     * @return the new dataset
     * @exception Exception if the weights array is of the wrong
     * length or contains negative weights or 
     * any other errors related to exemplars.
     */
    public final Exemplars resampleWithWeights(Random random) throws Exception{

	double [] weights = new double[numExemplars()];
	boolean foundOne = false;
	for (int i = 0; i < weights.length; i++) {
	    weights[i] = exemplar(i).weight();
	    if (!Utils.eq(weights[i], weights[0])) {
		foundOne = true;
	    }
	}
	if (foundOne) {
	    return resampleWithWeights(random, weights);
	} else {
	    return new Exemplars(this);
	}
    }


  /**
   * Creates a new dataset of the same size using random sampling
   * with replacement according to the given weight vector. The
   * weights of the exemplars in the new dataset are set to one.
   * The length of the weight vector has to be the same as the
   * number of exemplars in the dataset, and all weights have to
   * be positive.
   *
   * @param random a random number generator
   * @param weights the weight vector
   * @return the new dataset
   * @exception Exception if the weights array is of the wrong
   * length or contains negative weights or 
   * any other errors related to exemplars.
   */
    public final Exemplars resampleWithWeights(Random random, 
					       double[] weights) throws Exception{
	int len = weights.length;
	if (len != numExemplars()) {
	    throw new IllegalArgumentException("weights.length != numExemplars.");
	}
	Exemplars newData = new Exemplars(this, len);
	double[] probabilities = new double[len];
	double sumProbs = 0, sumOfWeights = Utils.sum(weights);
	for (int i = 0; i < len; i++) {
	    sumProbs += random.nextDouble();
	    probabilities[i] = sumProbs;
	}
	Utils.normalize(probabilities, sumProbs / sumOfWeights);
	
	// Make sure that rounding errors don't mess things up
	probabilities[len - 1] = sumOfWeights;
	int k = 0; int l = 0;
	sumProbs = 0;
	while ((k < len && (l < len))) {
	    if (weights[l] < 0) {
		throw new IllegalArgumentException("Weights have to be positive.");
	    }
	    sumProbs += weights[l];
	    while ((k < len) &&
		   (probabilities[k] <= sumProbs)) { 
		newData.m_Exemplars.addElement(new Exemplar(exemplar(l)));
		newData.exemplar(k).setWeight(1.0);
		k++;
	    }
	    l++;
	}
	return newData;
    }
    
    /**
     * Sets the relation's name.
     *
     * @param newName the new relation name.
     */
    public final void setRelationName(String newName) {
	
	m_RelationName = newName;
    }
    
    /**
     * Sorts the instances based on the ID attribute. For numeric attributes, 
     * instances are sorted in ascending order. For nominal attributes, 
     * instances are sorted based on the attribute label ordering 
     * specified in the header.  The instances inside an exemplar are not sorted.
     *
     */
    public final void sort() {
	
	int i,j;
	
	// move all instances with missing values to end
	j = numExemplars() - 1;
	i = 0;
	
	quickSort(0, j);
    }
    
    /**
     * Implements quicksort.
     *
     * @param lo0 the first index of the subset to be sorted
     * @param hi0 the last index of the subset to be sorted
     */
    private void quickSort( int lo0, int hi0) {
	
	int lo = lo0, hi = hi0;
	double mid, midPlus, midMinus;
	
	if (hi0 > lo0) {
	    
	    // Arbitrarily establishing partition element as the 
	    // midpoint of the array.
	    mid = exemplar((lo0 + hi0) / 2).idValue();
	    midPlus = mid + 1e-6;
	    midMinus = mid - 1e-6;
	    
	    // loop through the array until indices cross
	    while(lo <= hi) {
		
		// find the first element that is greater than or equal to 
		// the partition element starting from the left Index.
		while ((exemplar(lo).idValue() < 
			midMinus) && (lo < hi0)) {
		    ++lo;
		}
		
		// find an element that is smaller than or equal to
		// the partition element starting from the right Index.
		while ((exemplar(hi).idValue()  > 
			midPlus) && (hi > lo0)) {
		    --hi;
		}
		
		// if the indexes have not crossed, swap
		if(lo <= hi) {
		    swap(lo, hi);
		    ++lo;
		    --hi;
		}
	    }
	    
	    // If the right index has not reached the left side of array
	    // must now sort the left partition.
	    if(lo0 < hi) {
		quickSort(lo0,hi);
	    }
	    
	    // If the left index has not reached the right side of array
	    // must now sort the right partition.
	    if(lo < hi0) {
		quickSort(lo,hi0);
	    }
	}
    }
    

    /**
     * Swaps two instances in the set.
     *
     * @param i the first instance's index
     * @param j the second instance's index
     */
    private void swap(int i, int j){
	Exemplar tmp = new Exemplar((Exemplar)(m_Exemplars.elementAt(i)));
	m_Exemplars.setElementAt(new Exemplar((Exemplar)(m_Exemplars.elementAt(j))), i);
	m_Exemplars.setElementAt(tmp, j);
    }
    
    /**
     * Stratifies a set of exemplars according to its class values 
     * if the class attribute is nominal (so that afterwards a 
     * stratified cross-validation can be performed).
     *
     * @param numFolds the number of folds in the cross-validation
     * @exception UnassignedClassException if the class is not set
     */
    public final void stratify(int numFolds) {
	
	if (numFolds <= 0) {
	    throw new IllegalArgumentException("Number of folds must be greater than 1");
	}
	if (m_ClassIndex < 0) {
	    throw new UnassignedClassException("Class index is negative (not set)!");
	}
	if (classAttribute().isNominal()) {
	    
	    // sort by class
	    int index = 1;
	    while (index < numExemplars()) {
		Exemplar eg1 = exemplar(index - 1);
		for (int j = index; j < numExemplars(); j++) {
		    Exemplar eg2 = exemplar(j);
		    if (eg1.classValue() == eg2.classValue()) {
			swap(index,j);
			index++;
		    }
		}
		index++;
	    }
	    stratStep(numFolds);
	}
    }

    /**
     * Help function needed for stratification of set.
     *
     * @param numFolds the number of folds for the stratification
     */
    private void stratStep (int numFolds){
	
	Vector newExm = new Vector(m_Exemplars.size());
	int start = 0, j;
	
	// create stratified batch
	while(newExm.size() < numExemplars()){
	    j = start;
	    while (j < numExemplars()) {
		newExm.addElement(new Exemplar(exemplar(j)));
		j = j + numFolds;
	    }
	    start++;
	}
	
	m_Exemplars = newExm;
    }    
    
    /**
     * Computes the sum of all the exemplars' weights.
     *
     * @return the sum of all the exemplars' weights as a double
     */
    public final double[] sumsOfWeights() {
	
	double[] sum = new double[numExemplars()];
	
	for (int i = 0; i < numExemplars(); i++) {
	    sum[i] = exemplar(i).getInstances().sumOfWeights();
	}
	return sum;
    }
    
    /**
     * Creates the test set for one fold of a cross-validation on 
     * the dataset.
     *
     * @param numFolds the number of folds in the cross-validation. Must
     * be greater than 1.
     * @param numFold 0 for the first fold, 1 for the second, ...
     * @return the test set as a set of weighted instances
     * @exception Exception if the number of folds is less than 2
     * or greater than the number of exemplars
     * or any other errors related to exemplar occur
     */
    public Exemplars testCV(int numFolds, int numFold) throws Exception {
	
	int numExamForFold, first, offset;
	Exemplars test;
	
	if (numFolds < 2) {
	    throw new IllegalArgumentException("Number of folds must be at least 2!");
	}
	if (numFolds > numExemplars()) {
	    throw new IllegalArgumentException("Can't have more folds than Exemplars!");
	}
	numExamForFold = numExemplars() / numFolds;
	if (numFold < numExemplars() % numFolds){
	    numExamForFold++;
	    offset = numFold;
	}else
	    offset = numExemplars() % numFolds;
	
	first = numFold * (numExemplars() / numFolds) + offset;

	test = new Exemplars(this, numExamForFold);
	for(int i = 0; i < numExamForFold; i++)
	    test.m_Exemplars.addElement(new Exemplar((Exemplar)m_Exemplars.elementAt(first+i)));
	
	return test;
    }
    
    /**
     * Returns the exemplars as a string. 
     * It only shows each exemplar's ID value, class value and weight
     * as well as the ARFF header of the dataset 
     *
     * @return the set of exemplars as a string
     */
    public final String toString() {
	
	StringBuffer text = new StringBuffer();
	text.append("@relation " + Utils.quote(m_RelationName) + "\n\n");
	for (int i = 0; i < m_Attributes.length; i++) {
	    text.append(m_Attributes[i]);
	    if(i==m_IdIndex)
		text.append(" (ID Attribute)");
	    else if(i==m_ClassIndex)
		text.append(" (Class Attribute)");
	    
	    //text.append("\n");
	}
	
	Attribute id = idAttribute();
	Attribute cl = classAttribute();
	text.append("\n@Exemplars: \nID(" + id.name() + "); Class(" + 
		    cl.name() + "); Weight; sumOfInstances'Weights\n");
	
	double[] weights = sumsOfWeights();
	for(int j=0; j < m_Exemplars.size(); j++){
	    Exemplar eg = (Exemplar)m_Exemplars.elementAt(j);
	    text.append(id.value((int)eg.idValue())+"; "+
			cl.value((int)eg.classValue())
			+"; "+ eg.weight() + "; "+weights[j]+"\n");
	}
	text.append("There are totally "+numExemplars()
		    +" exemplars");
	return text.toString();
    }



    /**
     * Creates the training set skipping for one fold of a cross-validation 
     * on the exemplar set.
     *
     * @param numFolds the number of folds in the cross-validation. Must
     * be greater than 1.
     * @param numFold 0 for the first fold, 1 for the second, ...
     * @return the training set as a set of weighted 
     * instances
     * @exception Exception if the number of folds is less than 2
     * or greater than the number of exemplars or
     * or any other errors related to exemplar occur.
     */
    public Exemplars trainCV(int numFolds, int numFold) throws Exception{
	
	int numExamForFold, first, offset;
	Exemplars train;
	
	if (numFolds < 2)
	    throw new IllegalArgumentException
		("Number of folds must be at least 2!");
	
	if (numFolds > numExemplars())
	    throw new IllegalArgumentException
		("Can't have more folds than exemplars!");
	
	numExamForFold = numExemplars() / numFolds;
	if (numFold < numExemplars() % numFolds){
	    numExamForFold++;
	    offset = numFold;
	}else
	    offset = numExemplars() % numFolds;
	
	first = numFold * (numExemplars() / numFolds) + offset;
	train = new Exemplars(this, numExemplars() - numExamForFold);
	
	for(int i = 0; i < first; i++)
	    train.m_Exemplars.addElement(new Exemplar((Exemplar)m_Exemplars.elementAt(i)));
	
	for(int i = first; i < numExemplars()-numExamForFold; i++)
	    train.m_Exemplars.addElement(new Exemplar((Exemplar)m_Exemplars.elementAt(numExamForFold+i)));
	
	return train;
    }    
    

  /**
   * Creates the training set for one fold of a cross-validation 
   * on the dataset. The data is subsequently randomized based
   * on the given random number generator.
   *
   * @param numFolds the number of folds in the cross-validation. Must
   * be greater than 1.
   * @param numFold 0 for the first fold, 1 for the second, ...
   * @param random the random number generator
   * @return the training set 
   * @exception IllegalArgumentException if the number of folds is less than 2
   * or greater than the number of instances.
   */
  public Exemplars trainCV(int numFolds, int numFold, Random random) throws Exception {

    Exemplars train = trainCV(numFolds, numFold);
    train.randomize(random);
    return train;
  }
    
    /**
     * Main method for this class -- just performone run of 10-fold CV
     * and prints out the set.  Assume ID is the first attribute and class
     * is the last one.
     *
     * @param argv should contain one element: the name of an ARFF file
     */
    public static void main(String [] args) {
	
	try {
	    Reader r = null;
	    if (args.length > 1) {
		throw (new Exception("Usage: Exemplers <filename>"));
	    } else if (args.length == 0) {
		r = new BufferedReader(new InputStreamReader(System.in));
	    } else {
		r = new BufferedReader(new FileReader(args[0]));
	    }
	    
	    Instances data = new Instances(r);
	    data.setClassIndex(data.numAttributes()-1);
	    Exemplars e = new Exemplars(data, 0);
	    System.out.println("\nOriginal whole data:\n" + e.toString());
	    Exemplars ex = new Exemplars(e);
	    e = new Exemplars(ex, ex.numExemplars());
	    for(int i=0; i < ex.numExemplars(); i++)
		e.add(ex.exemplar(i));
	    e.stratify(3);
	    System.out.println("\nWhole data after stratification:\n" 
			       + e.toString());
	   
	    e.sort();
	    System.out.println("\nWhole data after sorting by Exemplar #:\n" 
			       + e.toString());
	 
	    Random ran = new Random(System.currentTimeMillis());
	    e.randomize(ran);
	    System.out.println("\nWhole data after randomization:\n" 
			       + e.toString());
	
	    Exemplars egs = e.resample(ran);
	    System.out.println("\nResampled data:\n" + egs.toString());
	
	    Exemplars test = e.testCV(10,1);
	    System.out.println("\nTesting data\n" + test.toString());
	
	    Exemplars train = e.trainCV(10,1);
	    System.out.println("\nTraining data:\n" + train.toString());
	    
	} catch (Exception ex) {
	    System.err.println(ex.getMessage());
	}
    }
}

     


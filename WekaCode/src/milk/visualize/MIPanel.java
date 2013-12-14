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
 *    MIPanel.java
 *    Copyright (C) 2001 Xin XU
 *
 */

package milk.visualize;
import milk.classifiers.*;
import milk.core.*;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Utils;

import java.beans.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import weka.gui.visualize.*;

/**
 * This class implements a panel that allows user to select some
 * exemplars to show the visualization.  Currently the visualization
 * is either the geometric display or the distributional display
 * of the instances inside a selected exemplar. <p>
 * The GeomPanel and DistributionPanel will extend this class to get
 * the specific panel
 *
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.00 $
 */
public class MIPanel extends JPanel{
    protected Exemplars exemplars = null;
    protected ClassPanel cp;
    protected Instances insts;
    protected JPanel ip = new JPanel();
    protected JComboBox clas, id;
    
    // A clumsy way to both reuse ClassPanel and make this class
    // a bean
    protected FastVector colorList;
    protected double minC, maxC;

    protected int classIndex = -1, idIndex = -1;    
    protected PropertyChangeSupport change;
    protected Color [] m_DefaultColors = {Color.blue,
					  Color.red,
					  Color.green,
					  Color.cyan,
					  Color.pink,
					  new Color(255, 0, 255),
					  Color.orange,
					  new Color(255, 0, 0),
					  new Color(0, 255, 0),
					  Color.white};
   
    /** 
     * Constructor: it has 2 comboboxes to select Id and class of
     * exemplars respectively, and one ClassPanel to show the class
     * labels
     */
    public MIPanel(){
	change = new PropertyChangeSupport(this);
	colorList = new FastVector(10);
	for (int noa = colorList.size(); noa < 10; noa++) {
	    Color pc = m_DefaultColors[noa % 10];
	    int ija =  noa / 10;
	    ija *= 2; 
	    for (int j=0;j<ija;j++) {
		pc = pc.darker();
	    }
	    
	    colorList.addElement(pc);
	}
	
	cp = new ClassPanel();
	cp.setColours(colorList);
	cp.addRepaintNotify(this);
	cp.setEnabled(false);

	ip = new JPanel();
	clas = new JComboBox();
	id = new JComboBox();
	ip.setLayout(new GridLayout(1,2));
	ip.add(clas);
	ip.add(id);
	ip.setEnabled(false);
	clas.addItem("Select class index");
	id.addItem("Select ID index");
	clas.setSelectedIndex(0);
	id.setSelectedIndex(0);

	setLayout(new BorderLayout());
	add(cp, BorderLayout.SOUTH);
	add(ip, BorderLayout.NORTH);

	clas.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    if(clas.getSelectedIndex()>0)
			setClassIndex(clas.getSelectedIndex()-1);
		}
	    });
	id.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    if(id.getSelectedIndex()>0)
			setIdIndex(id.getSelectedIndex()-1);
		}
	    });	
    }
    
    public void setInstances(Instances data){
	clas.removeAllItems();
	id.removeAllItems();
	clas.addItem("Select class index");
	id.addItem("Select ID index");
	insts = new Instances(data);
	
	for(int x=0; x < insts.numAttributes(); x++){
	    Attribute next = insts.attribute(x);
	    String nomOrNum = next.isNominal()?"(Nominal)":"(Numeric)";
	    clas.addItem(next.name()+nomOrNum);
	    id.addItem(next.name()+nomOrNum);
	}
	
	idIndex = -1;
	classIndex = -1;
	cp.removeAll();
	cp.setOn(false);
	cp.setEnabled(false);
	cp.repaint();
	ip.setEnabled(true);
	clas.setSelectedIndex(0);
	id.setSelectedIndex(0);
	Exemplars old = null;
	if(exemplars != null)
	    old = new Exemplars(exemplars);
	exemplars = null;	
	change.firePropertyChange("exemplars", old, exemplars);
    }
    
    public void setClassIndex(int cl){
	classIndex = cl;
	if(insts != null){	 
	    insts.setClassIndex(classIndex);
	    cp.setInstances(insts);
	    cp.setCindex(classIndex);
	    cp.setEnabled(true);
	    cp.setOn(true);
	    cp.repaint();
	    
	    if(idIndex != -1){   
		try{
		    Exemplars old = exemplars;
		    exemplars = new Exemplars(insts, idIndex);
		    change.firePropertyChange("exemplars", old, exemplars);
		}catch(Exception e){
		    JOptionPane.showMessageDialog(this,
						  e.getMessage(),
						  "Wrong ID",
						  JOptionPane.ERROR_MESSAGE);
		    exemplars = null;
		}
	    }
	}
    }
    
    public void setIdIndex(int id){
	idIndex = id;
	if((insts != null) && (classIndex != -1)){
	    try{
		Exemplars old = exemplars;
		exemplars = new Exemplars(insts, idIndex);
		change.firePropertyChange("exemplars", old, exemplars);
	    }catch(Exception e){
		JOptionPane.showMessageDialog(this,
					      e.getMessage(),
					      "Wrong ID",
					       JOptionPane.ERROR_MESSAGE);
		exemplars = null;
	    }
	}
    }
   
    /**
     * Functions provided only for bean box 
     */
    public Exemplars getExemplars(){
	return exemplars;
    }

    /**
     * Render this component.  
     * Since when the class color changes, the ClassPanel will call the
     * this function, this function then fires the property change event
     * to inform the change of colors
     *
     * @param gx the graphics context
     */
    public void paintComponent(Graphics gx){
	super.paintComponent(gx);
	if(exemplars != null){
	    if(exemplars.classAttribute().isNominal())
		change.firePropertyChange("colorList", null, colorList);
	    else{
		change.firePropertyChange("minC", null, new Double(minC));
		change.firePropertyChange("maxC", null, new Double(maxC));
	    }
	}
    }
    
    /**
     * Add the property change listener to hear the change of color and
     * "exemplars" property of this class.  The latter property specifies
     * the total exemplars read from the data file -- it's to be listened
     * by SelectPanel 
     * 
     * @param pcl the property change listener
     */
    public void addPropertyChangeListener(PropertyChangeListener pcl){
	change.addPropertyChangeListener("colorList", pcl);
	change.addPropertyChangeListener("maxC", pcl);
	change.addPropertyChangeListener("minC", pcl);
	change.addPropertyChangeListener("exemplars", pcl);
    }
    
    /**
     * Remove the property change listener to hear the change of color and
     * "exemplars" property of this class.  The latter property specifies
     * the total exemplars read from the data file -- it's to be listened
     * by SelectPanel 
     * 
     * @param pcl the property change listener
     */
    public void removePropertyChangeListener(PropertyChangeListener pcl){
	change.removePropertyChangeListener("colorList", pcl);
	change.removePropertyChangeListener("maxC", pcl);
	change.removePropertyChangeListener("minC", pcl);
	change.removePropertyChangeListener("exemplars", pcl);
    }
}

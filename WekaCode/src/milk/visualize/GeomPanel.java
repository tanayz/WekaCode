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
 *    GeomPanel.java
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

import weka.gui.visualize.*;
import java.beans.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;


/**
 * This class implements a panel that allows user to select some
 * exemplars to show their geometric distribution in 2D space
 *
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.00 $
 */
public class GeomPanel extends MIPanel {
    private Axis xIndex = new Axis(), yIndex= new Axis();
    
    private JComboBox xAttr = new JComboBox(), yAttr = new JComboBox();
    private SelectPanel select = new SelectPanel();
    private MIPlot2D plot = new MIPlot2D();	   
  
    /**
     * Constructor: besides the ones in MIPanel, it has two more
     * comboxes for obtaining the x and y axes
     * It also sets up SelectPanel and plot space 
     */  
    public GeomPanel(){
	super();
	xAttr.addItem("Select x-axis attribute");
	xAttr.setSelectedIndex(0);
	yAttr.addItem("Select y-axis attribute");
	yAttr.setSelectedIndex(0);
	ip.setLayout(new GridLayout(2,2));
	ip.add(clas);
	ip.add(id);
	ip.add(xAttr);
	ip.add(yAttr);
	
	xAttr.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    if(xAttr.getSelectedIndex() > 0){
			int index = xAttr.getSelectedIndex()-1;			
			setXIndex(index);
		    }
		}
	    });
	
	yAttr.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    if(yAttr.getSelectedIndex() > 0){
			int index = yAttr.getSelectedIndex()-1;			
			setYIndex(index);
		    }
		}
	    });
	
	addPropertyChangeListener(select);
	addPropertyChangeListener(plot);
	select.addPropertyChangeListener(plot);
	
	add(plot, BorderLayout.CENTER);
	add(select, BorderLayout.EAST);

	repaint();
    }    
     
    /**
     * Set the y axis, then it fire the property change event
     * to inform (the MIPlot) that the y axis changes
     *
     * @param index the attribute index on y axis
     */
    public void setYIndex(int index){
	Axis old = yIndex;
	yIndex = new Axis();
	yIndex.index = index;
	if(insts != null){
	    yIndex.minValue = yIndex.maxValue
		= insts.firstInstance().value(yIndex.index);
	    for(int y=1; y < insts.numInstances(); y++){
		Instance ins = insts.instance(y);
		if(ins.value(yIndex.index) > yIndex.maxValue)
		    yIndex.maxValue = ins.value(yIndex.index);
		else if(ins.value(yIndex.index) < yIndex.minValue)
		    yIndex.minValue = ins.value(yIndex.index);
	    }
	    
	    change.firePropertyChange("YAxis", old, yIndex);  
	}
    }
    
    public Axis getYIndex(){
	return yIndex;
    }   
    
    /**
     * Set the x axis, then it fire the property change event
     * to inform (the MIPlot) that the x axis changes
     *
     * @param index the attribute index on x axis
     */
    public void setXIndex(int index){
	Axis old = xIndex;
	xIndex = new Axis();
	xIndex.index = index;
	if(insts != null){
	    xIndex.minValue = xIndex.maxValue
		= insts.firstInstance().value(xIndex.index);
	    for(int y=1; y < insts.numInstances(); y++){
		Instance ins = insts.instance(y);
		if(ins.value(xIndex.index) > xIndex.maxValue)
		    xIndex.maxValue = ins.value(xIndex.index);
		else if(ins.value(xIndex.index) < xIndex.minValue)
		    xIndex.minValue = ins.value(xIndex.index);
	    }
		
	    change.firePropertyChange("XAxis", old, xIndex);  
	}
    }
    
    public Axis getXIndex(){
	return xIndex;
    }   

    public void setInstances(Instances data){
	super.setInstances(data);
	xAttr.removeAllItems();
	yAttr.removeAllItems();
	xAttr.addItem("Select x-axis attribute");
	yAttr.addItem("Select y-axis attribute");

	for(int x=0; x < insts.numAttributes(); x++){
	    Attribute next = insts.attribute(x);
	    String nomOrNum = next.isNominal()?"(Nominal)":"(Numeric)";
	    xAttr.addItem(next.name()+nomOrNum);
	    yAttr.addItem(next.name()+nomOrNum);
	}
	xAttr.setSelectedIndex(0);
	yAttr.setSelectedIndex(0);
	change.firePropertyChange("XAxis", xIndex, null);
	xIndex = new Axis();
	change.firePropertyChange("YAxis", yIndex, null);
	yIndex = new Axis();
    }

    /**
     * Add the property change listener to hear the change of "XAxis" and
     * "YAxis" property of this class.  The former property specifies the 
     * selected x axis and latter one specifies the y axis.
     * 
     * @param pcl the property change listener
     */
    public void addPropertyChangeListener(PropertyChangeListener pcl){
	super.addPropertyChangeListener(pcl);
	change.addPropertyChangeListener("XAxis", pcl);
	change.addPropertyChangeListener("YAxis", pcl);	
    }
    
    /**
     * Remove the property change listener to hear the change of "XAxis" and
     * "YAxis" property of this class.  The former property specifies the 
     * selected x axis and latter one specifies the y axis.
     * 
     * @param pcl the property change listener
     */
    public void removePropertyChangeListener(PropertyChangeListener pcl){
	super.removePropertyChangeListener(pcl);
	change.removePropertyChangeListener("XAxis", pcl);
	change.removePropertyChangeListener("YAxis", pcl);	
    }
}

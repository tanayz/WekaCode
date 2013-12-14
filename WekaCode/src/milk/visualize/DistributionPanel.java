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
 *    DistributionPanel.java
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

import weka.gui.visualize.ClassPanel;
import java.beans.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;


/**
 * This class implements a panel that allows user to select some
 * exemplars to show their distributions
 *
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.00 $
 */
public class DistributionPanel extends MIPanel {
    private double stdDev=0.0;  
    private Axis xIndex = new Axis();

    private JComboBox xAttr = new JComboBox();
    private JTextField sd = new JTextField();
    private JLabel sdlbl = new JLabel("Std. Deviations: ");
    private JPanel sdpnl= new JPanel(new BorderLayout());
    private SelectPanel select = new SelectPanel();
    private PlotPanel plot = new PlotPanel();	   
    
    /**
     * Constructor: besides the ones in MIPanel, it has two more
     * comboxes for obtaining the x axis and the std. dev.
     * It also sets up SelectPanel and plot space 
     */  
    public DistributionPanel(){
	super();
	xAttr.addItem("Select x-axis attribute");
	xAttr.setSelectedIndex(0);
	ip.setLayout(new GridLayout(2,2));
	ip.add(clas);
	ip.add(id);
	ip.add(xAttr);
	sdpnl.add(sdlbl, BorderLayout.WEST);
	sdpnl.add(sd, BorderLayout.CENTER);
	ip.add(sdpnl);	
	xAttr.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    if(xAttr.getSelectedIndex() > 0){
			int index = xAttr.getSelectedIndex()-1;	
			setXIndex(index);
		    }
		}
	    });
	
	sd.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    try{
			double sdv = Double.parseDouble(sd.getText());
			setStdDev(sdv);
		    }catch(Exception e){
			JOptionPane.showMessageDialog(null,
						      e.getMessage(),
						      "Wrong Std. Dev.",
						      JOptionPane.ERROR_MESSAGE);
		    }
		}
	    });	

	sd.setText(Utils.doubleToString(stdDev, 6));
	addPropertyChangeListener(select);
	addPropertyChangeListener(plot);
	select.addPropertyChangeListener(plot);
	
	add(plot, BorderLayout.CENTER);
	add(select, BorderLayout.EAST);

	repaint();
    }    
    
    public void setStdDev(double std){
	double old = stdDev;
	stdDev = std;
	sd.setText(Utils.doubleToString(stdDev, 6));
	change.firePropertyChange("SD", new Double(old), new Double(stdDev));
    }
    
    public double getStdDev(){
	return stdDev;
    }

    public void setXIndex(int index){
	Axis old = xIndex;
	xIndex = new Axis();
	xIndex.index = index;
	if(insts != null){
	    if(insts.attribute(index).isNominal())
		JOptionPane.showMessageDialog(this,
					      "X Attribute cannot be nominal.",
					      "Wrong X Attribute",
					      JOptionPane.ERROR_MESSAGE);
	    else{
		xIndex.minValue = xIndex.maxValue
		    = insts.firstInstance().value(xIndex.index);
		for(int y=1; y < insts.numInstances(); y++){
		    Instance ins = insts.instance(y);
		    if(ins.value(xIndex.index) > xIndex.maxValue)
			xIndex.maxValue = ins.value(xIndex.index);
		    else if(ins.value(xIndex.index) < xIndex.minValue)
			xIndex.minValue = ins.value(xIndex.index);
		}
		
		double var = insts.variance(index);
		if(var>0.0){
		    setStdDev(Math.sqrt(var));
		}
		
		change.firePropertyChange("XAxis", old, xIndex);  
	    }
	}
    }
    
    public Axis getXIndex(){
	return xIndex;
    }   

    public void setInstances(Instances data){
	super.setInstances(data);
	xAttr.removeAllItems();
	xAttr.addItem("Select x-axis attribute");
	
	for(int x=0; x < insts.numAttributes(); x++){
	    Attribute next = insts.attribute(x);
	    String nomOrNum = next.isNominal()?"(Nominal)":"(Numeric)";
	    xAttr.addItem(next.name()+nomOrNum);
	}
	xAttr.setSelectedIndex(0);	
	change.firePropertyChange("XAxis", xIndex, null);
	xIndex = new Axis();
    }

    /**
     * Add the property change listener to hear the change of "XAxis" and
     * "SD" property of this class.  The former property specifies the 
     * selected x axis and latter one specifies the Std. Dev. to be used
     * when deriving distributions
     * 
     * @param pcl the property change listener
     */
    public void addPropertyChangeListener(PropertyChangeListener pcl){
	super.addPropertyChangeListener(pcl);
	change.addPropertyChangeListener("XAxis", pcl);
	change.addPropertyChangeListener("SD", pcl);	
    }
    

    /**
     * Remove the property change listener to hear the change of "XAxis" and
     * "SD" property of this class.  The former property specifies the 
     * selected x axis and latter one specifies the Std. Dev. to be used
     * when deriving distributions
     * 
     * @param pcl the property change listener
     */
    public void removePropertyChangeListener(PropertyChangeListener pcl){
	super.removePropertyChangeListener(pcl);
	change.removePropertyChangeListener("XAxis", pcl);
	change.removePropertyChangeListener("SD", pcl);	
    }
}

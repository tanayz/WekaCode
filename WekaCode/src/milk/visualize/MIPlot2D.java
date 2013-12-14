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
 *    MIPlot2D.java
 *    Copyright (C) 2001 WEKA, Xin Xu
 *
 */

package milk.visualize;
import milk.classifiers.*;
import milk.core.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

import weka.core.*;
import java.beans.*;
import java.util.*;
import weka.gui.visualize.*;

/**
 * This class is similar to the Plot2D class except that it also
 * draws the ID of exemplar on each instance point. <p>
 * It sets up one PlotData2D for each exemplar and plots all the
 * PlotData2D stored.
 *
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.00 $
 */
public class MIPlot2D extends Plot2D implements PropertyChangeListener{
    
    /** The instances to be plotted */
    protected Exemplars plotExemplars=null;    
      
    /** The x and y axis, max and min values are of all the exemplars */
    protected Axis x, y;
    
    /** Tempory variable to store maxC and minC in case the 
	super.determineBounds() finds wrong ones.*/
    private double maxC=Double.NaN, minC=Double.NaN;

    /** Constructor */
    public MIPlot2D() {
	super();
	super.m_backgroundColour = Color.white;
	this.setBackground(m_backgroundColour);
	super.m_axisColour = Color.black;
	m_colorList = null;
	setEnabled(false);
	setVisible(true);
    }
    
    /**
     * The handler function when property changes are informed.
     * there are lots of properties can be listened by this
     * class: the exemplars to be plotted, the colors
     * from the ClassPanel, x and y axis value and index.
     * Only all these parameters are set can the geometric display
     * be plotted
     *
     * @param pce the property change event
     */ 
    public void propertyChange(PropertyChangeEvent pce){
	try{
	    // From SelectPanel
	    if(pce.getPropertyName().equals("addedExs"))
		setPlotExemplars((Exemplars)pce.getNewValue());
	}catch(Exception e){
	    JOptionPane.showMessageDialog(this,
					  e.getMessage(),
					  "Should never happen!",
					  JOptionPane.ERROR_MESSAGE);    
	}
	
	// From GoemPanel
	if(pce.getPropertyName().equals("colorList"))
	    setColours((FastVector)pce.getNewValue());    	
	
	if(pce.getPropertyName().equals("XAxis"))
	    setX((Axis)(pce.getNewValue()));       	
	
	if(pce.getPropertyName().equals("YAxis"))
	    setY((Axis)(pce.getNewValue()));       
	
	if(pce.getPropertyName().equals("maxC"))
	    setMaxC(((Double)pce.getNewValue()).doubleValue());
	
	if(pce.getPropertyName().equals("minC"))
	    setMinC(((Double)pce.getNewValue()).doubleValue());       	

	check();	
	repaint();
    }

    private void check(){
	if((x != null) && (y != null)){
	    if(plotExemplars != null){
		Attribute clas = plotExemplars.classAttribute();
		if(clas.isNominal() && (m_colorList != null))
		    setEnabled(true);
		else if((!Double.isNaN(maxC)) && (!Double.isNaN(minC)))
		    setEnabled(true);
	    }    
	    else
		setEnabled(true);
	}
    }   
    
    /**
     * Set the index and value of the attribute to go on the x axis<br>
     * Before it's stored in m_xIndex, m_maxX and m_minX, it's stored
     * in the tempory variable x.  This treatment is to walk around
     * the super.determineBounds() function, which will incorrectly 
     * set up the bounds in this context.
     *
     * @param anx the index of the attribute to use on the x axis
     */
    public void setX(Axis anx){
	x=anx;
	if(x != null)
	    setXValue(x);
    }
    
    /**
     * Set the index and value of the attribute to go on the y axis<br>
     * Before it's stored in m_yIndex, m_maxY and m_minY, it's stored
     * in the tempory variable y.  This treatment is to walk around
     * the super.determineBounds() function, which will incorrectly 
     * set up the bounds in this context.
     *
     * @param ay the index of the attribute to use on the y axis
     */
    public void setY(Axis ay){
	y=ay;
	if(y != null)
	    setYValue(y);
    }
    
    private void setXValue(Axis anx) {
	m_xIndex = anx.index;
	m_maxX = anx.maxValue;
	m_minX = anx.minValue;
	if(m_plots != null){
	    for(int i=0; i < m_plots.size(); i++){
		PlotData2D pl = (PlotData2D)m_plots.elementAt(i);
		pl.setXindex(m_xIndex);
	    }
	}
	m_axisChanged = true;
    }
    
    private void setYValue(Axis ay) {	
	m_yIndex = ay.index;
	m_maxY = ay.maxValue;
	m_minY = ay.minValue;
	if(m_plots != null){
	    for(int i=0; i < m_plots.size(); i++){
		PlotData2D pl = (PlotData2D)m_plots.elementAt(i);
		pl.setYindex(m_yIndex);
	    }
	}
	m_axisChanged = true;
    }
    
    /**
     * Set the max color if the class is numeric<br>
     * Before it's stored in m_maxC, it's stored in the tempory variable 
     * maxC.  This treatment is to walk around the super.determineBounds()
     * function, which will incorrectly set up the bounds in this context.
     *
     * @param maxc max bound of color
     */
    public void setMaxC(double maxc) {
	maxC = maxc;	
    }  
    
    /**
     * Set the min color if the class is numeric<br>
     * Before it's stored in m_minC, it's stored in the tempory variable 
     * minC.  This treatment is to walk around the super.determineBounds()
     * function, which will incorrectly set up the bounds in this context.
     *
     * @param minc min bound of color
     */
    public void setMinC(double minc) {
	minC = minc;	
    }
    
    /**
     * Sets the plot vectors from a set of exemplars
     * @param added the exemplars
     * @exception exception Exception if plots could not be set
     */
    public void setPlotExemplars(Exemplars added) throws Exception {
	super.removeAllPlots();   // Set xIndex, yIndex and cIndex to 0	
	if(added == null){
	    plotExemplars = null;
	    setEnabled(false);
	    return;
	}

	plotExemplars = new Exemplars(added);	
	int cIndex = plotExemplars.classIndex();
	
	for(int i=0; i < plotExemplars.numExemplars(); i++){
	    Exemplar ex = plotExemplars.exemplar(i);
	    Instances insts = ex.getInstances();
	    PlotData2D tmp = new PlotData2D(insts);
	    int num = insts.numInstances();
	    int[] shapes = new int[num];
	    int[] sizes = new int[num];
	    for(int j=0; j < num; j++){
		shapes[j] = Plot2D.X_SHAPE;
		sizes[j] = Plot2D.DEFAULT_SHAPE_SIZE;	
	    }
	    
	    tmp.setShapeType(shapes);
	    tmp.setShapeSize(sizes);
	    tmp.m_useCustomColour = false;	
	    tmp.setCindex(cIndex);	
	    tmp.setPlotName(Integer.toString((int)ex.idValue()));
	    addPlot(tmp);	    // determineBound() involved
	}
	// Reset
	if(x != null)
	    setXValue(x);
	if(y != null)
	    setYValue(y);
	m_cIndex = cIndex;
	if(!Double.isNaN(maxC))
	    m_maxC = maxC;
	if(!Double.isNaN(minC))
	    m_minC = minC;
    }    
    
    /**
     * Renders this component
     * @param gx the graphics context
     */
    public void paintComponent(Graphics gx) {
	
	//if(!isEnabled())
	//    return;
	
	super.paintComponent(gx);
	
	if(plotExemplars != null){	
	    gx.setColor(m_axisColour);
	    // Draw the axis name
	    String xname = plotExemplars.attribute(m_xIndex).name(),
		yname = plotExemplars.attribute(m_yIndex).name();
	    gx.drawString(yname, m_XaxisStart+m_labelMetrics.stringWidth("M"), m_YaxisStart+m_labelMetrics.getAscent()/2+m_tickSize);
	    gx.drawString(xname, m_XaxisEnd-m_labelMetrics.stringWidth(yname)+m_tickSize, (int)(m_YaxisEnd-m_labelMetrics.getAscent()/2));
	    
	    // Draw points
	    Attribute classAtt = plotExemplars.classAttribute();
	    for (int j=0;j<m_plots.size();j++) {
		PlotData2D temp_plot = (PlotData2D)(m_plots.elementAt(j));
		Instances instances = temp_plot.getPlotInstances();
		
		StringTokenizer st = new StringTokenizer
		    (instances.firstInstance().stringValue(plotExemplars.idIndex()),"_");

		//////////////////// TLD stuff /////////////////////////////////
		/*
		double[] mu = new double[plotExemplars.numAttributes()],
		    sgm = new double[plotExemplars.numAttributes()];
		st.nextToken(); // Squeeze first element
		int p=0;
		while(p<mu.length){
		    if((p==plotExemplars.idIndex()) || (p==plotExemplars.classIndex()))
			p++;
		    if(p<mu.length){
			mu[p] = Double.parseDouble(st.nextToken());
			sgm[p] = Double.parseDouble(st.nextToken());
			p++;
		    }
		}
		Instance ins = instances.firstInstance();
		gx.setColor((Color)m_colorList.elementAt((int)ins.classValue()));
		double mux=mu[m_xIndex], muy=mu[m_yIndex],
		    sgmx=sgm[m_xIndex], sgmy=sgm[m_yIndex];
		double xs = convertToPanelX(mux-3*sgmx), xe = convertToPanelX(mux+3*sgmx),
		    xleng = Math.abs(xe-xs);
		double ys = convertToPanelY(muy+3*sgmy), ye = convertToPanelY(muy-3*sgmy),
		    yleng = Math.abs(ye-ys);
		// Draw oval
		gx.drawOval((int)xs,(int)ys,(int)xleng,(int)yleng);
		// Draw a dot
		gx.fillOval((int)convertToPanelX(mux)-2, (int)convertToPanelY(muy)-2, 4, 4);
		*/
		//////////////////// TLD stuff /////////////////////////////////
		
		//////////////////// instance-based stuff /////////////////////////////////
		/*
		  double[] core = new double[plotExemplars.numAttributes()],
		    range=new double[plotExemplars.numAttributes()];
		st.nextToken(); // Squeeze first element
		int p=0;
		while(p<range.length){
		    if((p==plotExemplars.idIndex()) || (p==plotExemplars.classIndex()))
			p++;
		    if(p<range.length)
			range[p++] = Double.parseDouble(st.nextToken());
		}

		p=0;
		while(st.hasMoreTokens()){
		    if((p==plotExemplars.idIndex()) || (p==plotExemplars.classIndex()))
			p++;
		    core[p++] = Double.parseDouble(st.nextToken());
		}

		Instance ins = instances.firstInstance();
		gx.setColor((Color)m_colorList.elementAt((int)ins.classValue()));
		double rgx=range[m_xIndex], rgy=range[m_yIndex];
		double x1 = convertToPanelX(core[m_xIndex]-rgx/2),
		    y1 = convertToPanelY(core[m_yIndex]-rgy/2),
		    x2 = convertToPanelX(core[m_xIndex]+rgx/2),
		    y2 = convertToPanelY(core[m_yIndex]+rgy/2),
		    x = convertToPanelX(core[m_xIndex]),
		    y = convertToPanelY(core[m_yIndex]);
		
		// Draw a rectangle
		gx.drawLine((int)x1, (int)y1, (int)x2, (int)y1);
		gx.drawLine((int)x1, (int)y1, (int)x1, (int)y2);
		gx.drawLine((int)x2, (int)y1, (int)x2, (int)y2);
		gx.drawLine((int)x1, (int)y2, (int)x2, (int)y2);
		
		// Draw a dot
		gx.fillOval((int)x-3, (int)y-3, 6, 6);

		// Draw string
		StringBuffer text =new StringBuffer(temp_plot.getPlotName()+":"+instances.numInstances());		
		gx.drawString(text.toString(), (int)x1, (int)y2+m_labelMetrics.getHeight());
		*/
		//////////////////// instance-based stuff /////////////////////////////////
		
		//////////////////// normal graph /////////////////////////////////
		
		// Paint numbers
		for (int i=0;i<instances.numInstances();i++) {
		    Instance ins = instances.instance(i);
		    if (!ins.isMissing(m_xIndex) && !ins.isMissing(m_yIndex)) {
			if(classAtt.isNominal())
			    gx.setColor((Color)m_colorList.elementAt((int)ins.classValue()));
			else{
			    double r = (ins.classValue() - m_minC) / (m_maxC - m_minC);
			    r = (r * 240) + 15;
			    gx.setColor(new Color((int)r,150,(int)(255-r)));
			}
			
			double x = convertToPanelX(ins.value(m_xIndex));
			double y = convertToPanelY(ins.value(m_yIndex));
			
			String id = temp_plot.getPlotName();
			gx.drawString(id, 
				      (int)(x - m_labelMetrics.stringWidth(id)/2), 
				      (int)(y + m_labelMetrics.getHeight()/2));
		    }
		}
			
		//////////////////// normal graph /////////////////////////////////   
	    }
	}
	
	//////////////////// TLD stuff /////////////////////////////////
	// Draw two Guassian contour with 3 stdDev
	// (-1, -1) with stdDev 1, 2
	// (1, 1) with stdDev 2, 1
	/*gx.setColor(Color.black);
	double mu=-1.5, sigmx, sigmy; // class 0
	if(m_xIndex == 1)
	    sigmx = 1;		    
	else
	    sigmx = 2;
	if(m_yIndex == 1)
	    sigmy = 1;		    
	else
	    sigmy = 2;
	
	double x1 = convertToPanelX(mu-3*sigmx), x2 = convertToPanelX(mu+3*sigmx),
	    xlen = Math.abs(x2-x1);
	double y1 = convertToPanelY(mu+3*sigmy), y2 = convertToPanelY(mu-3*sigmy),
	    ylen = Math.abs(y2-y1);
	// Draw heavy oval
	gx.drawOval((int)x1,(int)y1,(int)xlen,(int)ylen);
	gx.drawOval((int)x1-1,(int)y1-1,(int)xlen+2,(int)ylen+2);
	gx.drawOval((int)x1+1,(int)y1+1,(int)xlen-2,(int)ylen-2);
	// Draw a dot
	gx.fillOval((int)convertToPanelX(mu)-3, (int)convertToPanelY(mu)-3, 6, 6);

	mu=1.5; // class 1
	if(m_xIndex == 1)
	    sigmx = 1;		    
	else
	    sigmx = 2;
	if(m_yIndex == 1)
	    sigmy = 1;		    
	else
	    sigmy = 2;
	
	x1 = convertToPanelX(mu-3*sigmx);
	x2 = convertToPanelX(mu+3*sigmx);
	xlen = Math.abs(x2-x1);
	y1 = convertToPanelY(mu+3*sigmy);
	y2 = convertToPanelY(mu-3*sigmy);
	ylen = Math.abs(y2-y1);
	// Draw heavy oval
	gx.drawOval((int)x1,(int)y1,(int)xlen,(int)ylen);
	gx.drawOval((int)x1-1,(int)y1-1,(int)xlen+2,(int)ylen+2);
	gx.drawOval((int)x1+1,(int)y1+1,(int)xlen-2,(int)ylen-2);
	// Draw a dot
	gx.fillOval((int)convertToPanelX(mu)-3, (int)convertToPanelY(mu)-3, 6, 6);
	*/
	//////////////////// TLD stuff /////////////////////////////////

	//////////////////// instance-based stuff /////////////////////////////////
	/*
	// Paint a log-odds line: 1*x0+2*x1=0
	double xstart, xend, ystart, yend, xCoeff, yCoeff;
	if(m_xIndex == 1)
	    xCoeff = 1;	
	else
	    xCoeff = 2;	
	if(m_yIndex == 1)
	    yCoeff = 1;	
	else
	    yCoeff = 2;	
	
	xstart = m_minX;
	ystart = -xstart*xCoeff/yCoeff;
	if(ystart > m_maxY){
	    ystart = m_maxY;
	    xstart = -ystart*yCoeff/xCoeff;
	}	
	yend = m_minY;
	xend = -yend*yCoeff/xCoeff;
	if(xend > m_maxX){
	    xend = m_maxX;
	    yend = -xend*xCoeff/yCoeff;
	}

	// Draw a heavy line
	gx.setColor(Color.black);
	gx.drawLine((int)convertToPanelX(xstart), (int)convertToPanelY(ystart),
		    (int)convertToPanelX(xend), (int)convertToPanelY(yend));
	gx.drawLine((int)convertToPanelX(xstart)+1, (int)convertToPanelY(ystart)+1,
		    (int)convertToPanelX(xend)+1, (int)convertToPanelY(yend)+1);
	gx.drawLine((int)convertToPanelX(xstart)-1, (int)convertToPanelY(ystart)-1,
		    (int)convertToPanelX(xend)-1, (int)convertToPanelY(yend)-1);
	*/
	//////////////////// instance-based stuff /////////////////////////////////
    }  
}

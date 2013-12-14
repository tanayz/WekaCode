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
 *    PlotPanel.java
 *    Copyright (C) 2001 WEKA, Xin XU
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

/**
 * This class implements a panel that display the distributions of the
 * selected exemplars.  The distributions are derived by summing the
 * small Gaussian on each instance within the exemplar.  Details see
 * Detterich et al's MI paper.
 *
 * @author WEKA
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.00 $
 */
public class PlotPanel extends JPanel implements PropertyChangeListener{

    /** precision constant */
    public static int MAX_PRECISION=10;
    
    /** Default colour for the axis */
    private Color m_axisColour = Color.black;
    
    /** Default colour for the plot background */
    private Color m_backgroundColour = Color.white;     
    
    /** The exemplars to be plotted */
    protected Exemplars plotExemplars=null;    
    
    /** The list of the colors used */
    protected FastVector colorList;
    
    /** The max and min color */
    private double m_minC = Double.NaN, m_maxC = Double.NaN;
    
    /** Indexes of the attributes to go on the x axis */
    protected Axis x=null;
    
    /** The std. deviations used in deriving the distributions */
    protected double stdDev = 0.0;

    /** The maximal value in Y axis */
    protected double maxY = 1.0;
    
    /** Axis padding */
    private final int m_axisPad = 5;
    
    /** Tick size */
    private final int m_tickSize = 5;
    
    /** The offsets of the axes once label metrics are calculated */
    private int m_XaxisStart=0;
    private int m_XaxisEnd=0;    
    private int m_YaxisStart=0;
    private int m_YaxisEnd=0;
    
    /** Whether the distribution curve is up or down */
    protected boolean isUp = true;
         
    /** Font for labels */
    private Font m_labelFont;
    private FontMetrics m_labelMetrics=null; 
    
    /** Constructor */
    public PlotPanel() {
	setBackground(m_backgroundColour);	
	setEnabled(false);
    }
    
    /**
     * The handler function when property changes are informed.
     * there are lots of properties can be listened by this
     * class: the exemplars to be plotted, the colors
     * from the ClassPanel, x axis value and index, and std. dev.
     * Only all these parameters are set can the distribution be
     * plotted
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
	// From DistributionPanel
	if(pce.getPropertyName().equals("colorList"))
	    setColours((FastVector)pce.getNewValue());
	
	if(pce.getPropertyName().equals("maxC"))
	    setMaxC(((Double)pce.getNewValue()).doubleValue());
	
	if(pce.getPropertyName().equals("minC"))
	    setMinC(((Double)pce.getNewValue()).doubleValue());       	
	
	if(pce.getPropertyName().equals("XAxis"))
	    setXValue((Axis)(pce.getNewValue()));       	
	
	if(pce.getPropertyName().equals("SD"))
	    setStdDev(((Double)pce.getNewValue()).doubleValue());       		
	
	check();
	repaint();
    }

    public void setStdDev(double sd){
	stdDev = sd;
	maxY = 1.0 / (stdDev * Math.sqrt(2.0*Math.PI));
    }
    
    private void check(){
	if(x != null){
	    if(plotExemplars != null){
		Attribute clas = plotExemplars.classAttribute();
		if(clas.isNominal() && (colorList != null))
		    setEnabled(true);
		else if((!Double.isNaN(m_maxC)) && (!Double.isNaN(m_minC)))
		    setEnabled(true);
	    }    
	}
    }   
    
    /**
     * Set a list of colours to use when colouring points according
     * to class values or cluster numbers
     * @param cols the list of colours to use
     */
    public void setColours (FastVector cols) {
	colorList = cols;
    }
    
    public void setMaxC(double maxc) {
	m_maxC = maxc;
    }
    
    public void setMinC(double minc) {
	m_minC = minc;
    }
    
    /**
     * Set the index of the attribute to go on the x axis
     * @param x the index of the attribute to use on the x axis
     */
    public void setXValue(Axis anx) {
	x = anx;
    }

    /** 
     * Return the current max value of the attribute plotted on the x axis
     * @return the x value
     */
    public Axis getXValue() {
	return x;
    }
    
    /** 
     * Return the current max value of the attribute plotted on the y axis,
     * i.e. 1/[sqrt(2*PI) * std. dev.]
     * @return the max y value
     */
    public double getMaxY() {
	return maxY;
    }
    
    /**
     * Sets the exemplars to be plotted
     * @param added the exemplars
     * @exception exception Exception if exemplars could not be set
     */
    public void setPlotExemplars(Exemplars added) throws Exception {
	if(added != null)
	    plotExemplars = new Exemplars(added);
	else{
	    setEnabled(false);
	    plotExemplars = null;
	}
    }
    
    /**
     * Set up fonts and font metrics
     * @param gx the graphics context
     */
    private void setFonts(Graphics gx) {
	if (m_labelMetrics == null) {
	    m_labelFont = new Font("Monospaced", Font.PLAIN, 12);
	    m_labelMetrics = gx.getFontMetrics(m_labelFont);
	}
	gx.setFont(m_labelFont);
    }
    
    /**
     * convert a Panel x coordinate to a raw x value.
     * @param scx The Panel x coordinate
     * @return A raw x value.
     */
    public double convertToAttribX(double scx) {
	double temp = m_XaxisEnd - m_XaxisStart;
	double temp2 = ((scx - m_XaxisStart)*(x.maxValue - x.minValue))/temp 
	    + x.minValue;
	
	return temp2;
    }
    
    /**
     * convert a Panel y coordinate to a raw y value.
     * @param scy The Panel y coordinate
     * @return A raw y value.
     */
    public double convertToAttribY(double scy) {
	double temp = m_YaxisEnd - m_YaxisStart;
	double temp2 = ((m_YaxisEnd - scy) * maxY) / temp;
	
	return temp2;
    }
    
    /**
     * Convert an raw x value to Panel x coordinate.
     * @param xval the raw x value
     * @return an x value for plotting in the panel.
     */
    public double convertToPanelX(double xval) {
	double temp = (xval - x.minValue)/(x.maxValue - x.minValue);
	double temp2 = temp * (m_XaxisEnd - m_XaxisStart);
	
	temp2 = temp2 + m_XaxisStart;
	
	return temp2;
    }
    
  
    /**
     * Convert an raw y value to Panel y coordinate.
     * @param yval the raw y value
     * @return an y value for plotting in the panel.
     */
    public double convertToPanelY(double yval) {
	double temp = yval/maxY;
	double temp2 = temp * (m_YaxisEnd - m_YaxisStart);
	
	temp2 = m_YaxisEnd - temp2;
	
	return temp2;
    }
   
  /**
   * Draws the distribution
   * @param gx the graphics context
   */
  private void paintData(Graphics gx) {
      if((plotExemplars == null) || (x == null))
	  return;
      
      setFonts(gx);
      int w = this.getWidth();
      boolean[] xlabels = new boolean[w];

      Attribute classAtt = plotExemplars.classAttribute(),
	  id = plotExemplars.idAttribute();
      int hf = m_labelMetrics.getAscent();
      
      for(int i=0; i < plotExemplars.numExemplars(); i++){
	  Exemplar ex = plotExemplars.exemplar(i);
	  if(classAtt.isNominal())
	      gx.setColor((Color)colorList.elementAt((int)ex.classValue()));
	  else{
	      double r = (ex.classValue() - m_minC) / (m_maxC - m_minC);
	      r = (r * 240) + 15;
	      gx.setColor(new Color((int)r,150,(int)(255-r)));
	  }
	  
	  double preY=0;
	  for(int j=0; j < ex.getInstances().numInstances(); j++){
	      Instance ins = ex.getInstances().instance(j);
	      double xValue = convertToAttribX(m_XaxisStart);
	      double tmp = - (xValue - ins.value(x.index)) *
		  (xValue - ins.value(x.index)) / 
		  (2.0 * stdDev * stdDev);
	      preY += Math.exp(tmp) * ins.weight();
	  }
	  preY /= ex.getInstances().sumOfWeights();
	  preY *= maxY;
	 	  
	  for(int k=m_XaxisStart+1; k < m_XaxisEnd; k++){
	      double currY=0;
	      for(int l=0; l < ex.getInstances().numInstances(); l++){
		  Instance ins = ex.getInstances().instance(l);
		  double xValue = convertToAttribX(k);
		  double tmp = - (xValue - ins.value(x.index)) *
		      (xValue - ins.value(x.index)) / 
		      (2.0 * stdDev * stdDev);
		  currY += Math.exp(tmp) * ins.weight();
	      }
	      currY /= ex.getInstances().sumOfWeights();
	      currY *= maxY;
	      
	      // Draw the distribution	      
	      int plotPreY = (int)convertToPanelY(preY);
	      int plotCurrY = (int)convertToPanelY(currY);
	      gx.drawLine(k-1, plotPreY, k, plotCurrY);	 
     
	      // If peak or valley appears, specify the x value
	      if(isUp && (preY > currY)){
		  Font old = gx.getFont();
		  gx.setFont(new Font("Monospaced", Font.PLAIN, 10));
		  String idvalue = Integer.toString((int)ex.idValue());
		  gx.drawString(idvalue, 
				k-1 - m_labelMetrics.stringWidth(idvalue)/2, 
				plotCurrY);
		  xlabels[k-1] = true;
		  gx.setFont(old);
	      }	      
	      isUp = (currY >= preY);      
	      preY = currY;	      
	  }
      }

      // Draw the number labels on x axis where various peaks gather
      gx.setColor(m_axisColour);
      int start=0, end=0;
      while(start < w){
	  if(xlabels[start]){
	      end = start;
	      int falseCount = 0;
	      while(end < w){
		  while(xlabels[end++]);  // Find the first false from start
		  int m=end;
		  // Count the number of falses
		  for(falseCount=0; (m < xlabels.length) && (!xlabels[m]); m++, falseCount++);

		  if((falseCount < 28) && (m < xlabels.length)) end = m;
		  else  break;
	      }
	      if(!xlabels[end])
		  --end;

	      int avg = (start+end)/2;
	      double xValue = convertToAttribX(avg);	    	    
	      String stringX = Utils.doubleToString(xValue, 1);
	      Font old = gx.getFont();
	      gx.setFont(new Font("Monospaced", Font.PLAIN, 10));		
	      gx.drawString(stringX, 
			    avg-m_labelMetrics.stringWidth(stringX)/2,
			    m_YaxisEnd+hf+m_tickSize);
	      gx.drawLine(avg, m_YaxisEnd, avg, m_YaxisEnd+m_tickSize);
	      gx.setFont(old);
	      start = end;
	  }
	  else
	      ++start;
      }
  }

    
    /**
     * Draws the axis 
     * @param gx the graphics context
     */
    private void paintAxis(Graphics gx) {
	setFonts(gx);
	gx.setColor(m_axisColour);
	int mxs = m_XaxisStart;
	int mxe = m_XaxisEnd;
	int mys = m_YaxisStart;
	int mye = m_YaxisEnd;
	
	int h = this.getHeight();
	int w = this.getWidth();
	int hf = m_labelMetrics.getAscent();
	int mswx=0;
	int mswy=0;
	
	int precisionXmax = 1;
	int precisionXmin = 1;
	int precisionYmax = 1;

	String minStringX="", maxStringX ="";
	int whole, nondecimal;
	double decimal;
	if(x != null){	
	    whole = (int)Math.abs(x.maxValue);
	    decimal = Math.abs(x.maxValue) - whole;
	    nondecimal = (whole > 0) 
		? (int)(Math.log(whole) / Math.log(10))
		: 1;
	    
	    precisionXmax = (decimal > 0) 
		? (int)Math.abs(((Math.log(Math.abs(x.maxValue)) / 
				  Math.log(10))))+2
		: 1;
	    if (precisionXmax > MAX_PRECISION) {
		precisionXmax = 1;
	    }
	    
	    maxStringX = Utils.doubleToString(x.maxValue,
					      nondecimal+1+precisionXmax,
					      precisionXmax);
	    
	    whole = (int)Math.abs(x.minValue);
	    decimal = Math.abs(x.minValue) - whole;
	    nondecimal = (whole > 0) 
		? (int)(Math.log(whole) / Math.log(10))
		: 1;
	    precisionXmin = (decimal > 0) 
		? (int)Math.abs(((Math.log(Math.abs(x.minValue)) / 
				  Math.log(10))))+2
		: 1;
	    if (precisionXmin > MAX_PRECISION) {
		precisionXmin = 1;
	    }
	    
	    minStringX = Utils.doubleToString(x.minValue,
					      nondecimal+1+precisionXmin,
					      precisionXmin);
	    
	    mswx = m_labelMetrics.stringWidth(maxStringX);
	}	   
	
	whole = (int)Math.abs(maxY);
	decimal = Math.abs(maxY) - whole;
	nondecimal = (whole > 0) 
	    ? (int)(Math.log(whole) / Math.log(10))
	    : 1;
	precisionYmax = (decimal > 0) 
	    ? (int)Math.abs(((Math.log(Math.abs(maxY)) / 
			      Math.log(10))))+2
	    : 1;
	if (precisionYmax > MAX_PRECISION) {
	    precisionYmax = 1;
	}
	
	String stringY = Utils.doubleToString(maxY, 
					      nondecimal+1+precisionYmax,
					      precisionYmax);
	
	mswy = m_labelMetrics.stringWidth(stringY);
	
	m_YaxisStart = m_axisPad;
	m_XaxisStart = 0+m_axisPad+m_tickSize+mswy;
	
	m_XaxisEnd = w-m_axisPad-(mswx/2);
	
	m_YaxisEnd = h-m_axisPad-(2 * hf)-m_tickSize;
	
	if(x != null){
	    if (w > (2 * mswx)) {	
		gx.drawString(maxStringX, 
			      m_XaxisEnd-(mswx/2),
			      m_YaxisEnd+hf+m_tickSize);
		
		mswx = m_labelMetrics.stringWidth(minStringX);
		gx.drawString(minStringX,
			      (m_XaxisStart-(mswx/2)),
			      m_YaxisEnd+hf+m_tickSize);
	    }     
	}
	
	// draw the y axis
	if (h > (2 * hf)) {
	    gx.drawString(stringY, 
			  m_XaxisStart-mswy-m_tickSize,
			  m_YaxisStart+(hf));
	    
	    gx.drawString("0",
			  (m_XaxisStart-mswy-m_tickSize),
			  m_YaxisEnd);
	}	
	
	gx.drawLine(m_XaxisStart,
		    m_YaxisStart,
		    m_XaxisStart,
		    m_YaxisEnd);
	gx.drawLine(m_XaxisStart,
		    m_YaxisEnd,
		    m_XaxisEnd,
		    m_YaxisEnd);	
    }
        
    
    /**
     * Renders this component
     * @param gx the graphics context
     */
    public void paintComponent(Graphics gx) {
	super.paintComponent(gx);
	if(!isEnabled())
	    return;
	
	paintAxis(gx);  
	paintData(gx);
	
    }  
}

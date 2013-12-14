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
 *    SelectPanel.java
 *    Copyright (C) 2001 Xin XU
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
 * This class implements two list controls which can be used to 
 * select exemplars to be shown on the panel.
 *
 * @author Xin XU (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.00 $
 */
public class SelectPanel extends JPanel 
    implements ListDataListener, PropertyChangeListener {
    private static Color listBg = Color.black, listFg = Color.white;
    private JList choose, added;
    private DefaultListModel chooseModel, addModel;
    
    private JButton add, rm, addAll, rmAll;
    private JPanel jp;
    private Exemplars exemplars = null;
    private Exemplars addedExs = null;
    private PropertyChangeSupport pcs;
    private FastVector colorList;
    private double m_minC = Double.NaN, m_maxC = Double.NaN;
    private JScrollPane chooseSP, addSP;
    
    /** 
     * Constructor: arrange the display */
    public SelectPanel(){	
	jp = new JPanel();
	jp.setLayout(new GridLayout(2,2));
	jp.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
	add = new JButton("Add");
	addAll = new JButton("Add All");
	rm = new JButton("Remove");
	rmAll = new JButton("Remove All");
	jp.add(add);
	jp.add(addAll);
	jp.add(rm);
	jp.add(rmAll);
	
	chooseModel = new DefaultListModel();
	addModel = new DefaultListModel();
	choose = new JList(chooseModel);
	added = new JList(addModel);	
	choose.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
	added.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
	chooseModel.addListDataListener(this);
	addModel.addListDataListener(this);
	
	chooseSP = new JScrollPane(choose);
	chooseSP.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
	addSP = new JScrollPane(added);	
	addSP.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
	pcs = new PropertyChangeSupport(this);	
	
	setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
	add(chooseSP);		
	add(jp);
	add(addSP);
	setBorder(BorderFactory.createEmptyBorder(10, 5, 10, 5));
	
	add.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    try{
			int[] indice = choose.getSelectedIndices();
			Exemplars old = new Exemplars(addedExs);

			for(int j=0, numRmved=0; j < indice.length; j++,numRmved++){
			    int index = indice[j] - numRmved;
			    Object tmp = chooseModel.remove(index);
			    addModel.addElement(tmp);
			    addedExs.add((Exemplar)tmp);
			}
			
			int size = chooseModel.getSize();		    
			if (size > 0)
			    choose.setSelectedIndex(choose.getSelectedIndex());		      
			pcs.firePropertyChange("addedExs", old, addedExs); 
		    }
		    catch(Exception e){
			System.out.println(e.getMessage());
			e.printStackTrace();
		    }
		}
	    });
	
	addAll.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    try{
			Exemplars old = new Exemplars(addedExs);
			while(chooseModel.getSize() > 0){
			    Object tmp = chooseModel.remove(chooseModel.getSize()-1);
			    addedExs.add((Exemplar)tmp);	
			    addModel.addElement(tmp);			
			}
			
			pcs.firePropertyChange("addedExs", old, addedExs); 
		    }
		    catch(Exception e){
			System.out.println(e.getMessage());
			e.printStackTrace();
		    }
		}
	    });		
	
	rm.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    int[] indice = added.getSelectedIndices();
		    Exemplars old = new Exemplars(addedExs);

		    for(int j=0, numRmved=0; j < indice.length; j++,numRmved++){
			int index = indice[j] - numRmved;
			Object tmp = addModel.remove(index);
			addedExs.delete(index);			
			chooseModel.addElement(tmp);			
		    }
		    
		    int size = addModel.getSize();		    
		    if (size > 0)
			added.setSelectedIndex(added.getSelectedIndex());
		    
		    pcs.firePropertyChange("addedExs", old, addedExs); 
		}
	    });
	
	rmAll.addActionListener(new ActionListener(){
		public void actionPerformed(ActionEvent ae){
		    Exemplars old = new Exemplars(addedExs);
		    while(addModel.getSize() > 0){	
			int size = addModel.getSize();
			Object tmp = addModel.remove(size-1);
			addedExs.delete(size-1);			
			chooseModel.addElement(tmp);			
		    }
		    
		    pcs.firePropertyChange("addedExs", old, addedExs); 
		}
	    });	
	
	setEnabled(false);
    }
    
    /**
     * The handler function when property changes are informed.
     * there are two kinds of properties can be listened by this
     * class: all the exemplars read from the data and the colors
     * from the ClassPanel
     *
     * @param pce the property change event
     */
    public void propertyChange(PropertyChangeEvent pce){
	if(pce.getPropertyName().equals("exemplars"))
	    setExemplars((Exemplars)pce.getNewValue());
	
	if(pce.getPropertyName().equals("colorList"))
	    setColors((FastVector)pce.getNewValue());
	
	if(pce.getPropertyName().equals("maxC"))
	    setMaxC(((Double)pce.getNewValue()).doubleValue());
	
	if(pce.getPropertyName().equals("minC"))
	    setMinC(((Double)pce.getNewValue()).doubleValue());       	
	
	check();
	repaint();
    }

    public void setColors(FastVector c){
	colorList = c;
    }    
    public void setMaxC(double maxc) {
	m_maxC = maxc;	
    }    
    public void setMinC(double minc) {
	m_minC = minc;	
    }
    
    private void check(){
	if(exemplars != null){
	    Attribute clas = exemplars.classAttribute();
	    if(clas.isNominal() && (colorList != null))
		setEnabled(true);
	    else if((!Double.isNaN(m_maxC)) && (!Double.isNaN(m_minC)))
		setEnabled(true);
	}    
    }   
    
    public void setExemplars(Exemplars exs){
	
	if(exs == null){ // Instances are updated
	    chooseModel.removeAllElements();
	    addModel.removeAllElements();	
	    exemplars = null;
	    Exemplars old = addedExs;
	    addedExs = null;
	    pcs.firePropertyChange("addedExs", old, addedExs); 
	    setEnabled(false);
	    return;
	}
	
	exemplars = new Exemplars(exs);
	for(int i=0; i < exemplars.numExemplars(); i++){
	    Exemplar one = exemplars.exemplar(i);
	    chooseModel.addElement(one);
	}
	if(addedExs == null)
	    addedExs = new Exemplars(exs, 0);				
    }
    
    public Exemplars getAddedExs(){
	return addedExs;
    }
    
    public void paintComponent(Graphics gx){	
	super.paintComponent(gx);	
	if(!isEnabled())
	    return;
		 
	ListCellRenderer lcr = new ListCellRenderer(){
		public Component getListCellRendererComponent
		    (JList list,
		     Object value,
		     int index,
		     boolean isSelected,
		     boolean cellHasFocus)
                {
		    Attribute id = exemplars.idAttribute(), 
			clas = exemplars.classAttribute();
		    JLabel jl = new JLabel();
		    jl.setOpaque(true);
		    Exemplar ex = (Exemplar)value;
		    String idv = Integer.toString((int)ex.idValue())+": ";
		    if(clas.isNominal()){
			jl.setText(idv+id.value((int)ex.idValue())+
				   "(class: "+
				   clas.value((int)ex.classValue())+
				   ")");
			Color fg = isSelected ? listFg : (Color)colorList.elementAt((int)ex.classValue());
			Color bg = isSelected ? listBg : Color.white;
			jl.setForeground(fg);
			jl.setBackground(bg);
		    }
		    else{
			jl.setText(idv+id.value((int)ex.idValue())+
				   "(class: "+
				   ex.classValue()+
				   ")");
			
			double r = (ex.classValue() - m_minC) / (m_maxC - m_minC);
			r = (r * 240) + 15;	
			Color fg = isSelected ? listFg : new Color((int)r,150,(int)(255-r));
			Color bg = isSelected ? listBg : Color.white;		 
			jl.setForeground(fg);
			jl.setBackground(bg);
		    }		  
		    return jl;
		}
	    };
	
	choose.setCellRenderer(lcr);
	added.setCellRenderer(lcr);			
    }

    public void intervalAdded(ListDataEvent e){ checkEmpty();}
    public void intervalRemoved(ListDataEvent e){ checkEmpty();}   
    public void contentsChanged(ListDataEvent e){ checkEmpty();}
    
    private void checkEmpty(){ 
	if (choose.getModel().getSize() == 0) {
	    //No item, disable add button.
	    add.setEnabled(false);
	    addAll.setEnabled(false);
	} 
	else{
	    add.setEnabled(true);
	    addAll.setEnabled(true);
	}
	
	if (added.getModel().getSize() == 0) {
	    //No item, disable rm button.
	    rm.setEnabled(false);
	    rmAll.setEnabled(false);
	} 
	else{
	    rm.setEnabled(true);
	    rmAll.setEnabled(true);
	}
    }
    
    /**
     * Add the property change listener to hear the change of "addedExs"
     * property of this class.  This property specifies the selected 
     * exemplars.
     * 
     * @param pcl the property change listener
     */
    public void addPropertyChangeListener(PropertyChangeListener pcl){
	pcs.addPropertyChangeListener("addedExs", pcl);
    }
    
    /**
     * Remove the property change listener to hear the change of "addedExs"
     * property of this class.  This property specifies the selected 
     * exemplars.
     * 
     * @param pcl the property change listener
     */
    public void removePropertyChangeListener(PropertyChangeListener pcl){
	pcs.removePropertyChangeListener("addedExs", pcl);
    }

    /**
     * Main function for testing, it simply display the two list controls 
     */
    public static void main(String [] args) {	
	try {
	    final JFrame jf = new JFrame("Weka Knowledge Explorer: Preprocess");
	    jf.getContentPane().setLayout(new BorderLayout());
	    final SelectPanel sp = new SelectPanel();
	    jf.getContentPane().add(sp, BorderLayout.CENTER);
	    jf.addWindowListener(new WindowAdapter() {
		    public void windowClosing(WindowEvent e) {
			jf.dispose();
			System.exit(0);
		    }
		});
	    jf.pack();
	    jf.setSize(100, 600);
	    jf.setVisible(true);
	} catch (Exception ex) {
	    ex.printStackTrace();
	    System.err.println(ex.getMessage());
	}
    }
}

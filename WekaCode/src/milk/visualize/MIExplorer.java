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
 *    MIExplorer.java
 *    Copyright (C) 2001 WEKA, Xin XU
 *
 */
package milk.visualize;
import milk.classifiers.*;
import milk.core.*;

import weka.core.Utils;
import weka.gui.*;
import weka.gui.explorer.PreprocessPanel;
import java.awt.*;
import java.awt.event.*;
import java.beans.*;
import javax.swing.*;

/**
 * This class puts all components together to visualize the 
 * MI data, i.e. it has a PreprocessPanel, a DistributionPanel
 * and a GeomPanel
 * 
 * @author authors of weka.gui.explorer.Explorer
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.0 $
 */
public class MIExplorer extends JPanel {
    
    /** The panel for preprocessing instances */
    protected PreprocessPanel m_PreprocessPanel = new PreprocessPanel();
    
    /** The panel to show distributions */
    protected DistributionPanel m_DistributionPanel = new DistributionPanel(); 

    /** The panel to show data geometrically */
    protected GeomPanel m_GeomPanel = new GeomPanel();
    
    /** The tabbed pane that controls which sub-pane we are working with */
    protected JTabbedPane m_TabbedPane = new JTabbedPane();
    
    /** The panel for log and status messages */
    protected LogPanel m_LogPanel = new LogPanel(new WekaTaskMonitor());
    
    
    /**
     * Creates the environment
     */
    public MIExplorer() {
	
	m_LogPanel.logMessage("MIExplorer");
	m_LogPanel.logMessage("(c) 2001 The University of Waikato, Hamilton,"
			      + " New Zealand");
	m_LogPanel.logMessage("web: http://www.cs.waikato.ac.nz/~ml/");
	m_LogPanel.logMessage("email: wekasupport@cs.waikato.ac.nz");
	m_LogPanel.statusMessage("Welcome to the Weka Knowledge Explorer");
	m_PreprocessPanel.setLog(m_LogPanel);
	
	m_TabbedPane.addTab("Preprocess", null, m_PreprocessPanel,
			    "Open/Edit/Save instances");
	m_TabbedPane.addTab("Distributions", null, m_DistributionPanel,
			    "Visualize Distributions");
	m_TabbedPane.addTab("2D visualization", null, m_GeomPanel,
			    "Visualize Data in 2D");	
	
	m_TabbedPane.setSelectedIndex(0);
	m_TabbedPane.setEnabledAt(1, false); 
	m_TabbedPane.setEnabledAt(2, false); 
	m_PreprocessPanel.addPropertyChangeListener(new PropertyChangeListener() {
		public void propertyChange(PropertyChangeEvent e) {
		    m_DistributionPanel.setInstances(m_PreprocessPanel
						     .getInstances());
		    m_GeomPanel.setInstances(m_PreprocessPanel
					     .getInstances());  
		    m_TabbedPane.setEnabledAt(1, true);
		    m_TabbedPane.setEnabledAt(2, true);
		}
	    });
	
	setLayout(new BorderLayout());
	add(m_TabbedPane, BorderLayout.CENTER);	
	add(m_LogPanel, BorderLayout.SOUTH);
    }
    
    /**
     * Tests out the explorer environment.
     *
     * @param args ignored.
     */
    public static void main(String [] args) {
	
	try {
	    MIExplorer explorer = new MIExplorer();
	  
	    final JFrame jf = new JFrame("MIExplorer");
	    jf.getContentPane().setLayout(new BorderLayout());
	    jf.getContentPane().add(explorer, BorderLayout.CENTER);

	    jf.addWindowListener(new WindowAdapter() {
		    public void windowClosing(WindowEvent e) {
			jf.dispose();
			System.exit(0);
		    }
		});
	    jf.pack();
	    jf.setSize(800, 600);
	    jf.setVisible(true);	  	
	} catch (Exception ex) {
	    ex.printStackTrace();
	    System.err.println(ex.getMessage());
	}
    }
}

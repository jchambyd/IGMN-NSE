/**
 * =============================================================================
 * Federal University of Rio Grande do Sul (UFRGS)
 * Connectionist Artificial Intelligence Laboratory (LIAC)
 * Jorge C. Chamby Diaz - jccdiaz@inf.ufrgs.br
 * =============================================================================
 * Copyright (c) 2017 Jorge C. Chamby Diaz, jchambyd at gmail dot com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * =============================================================================
 */
package liac.igmn.gui;

import java.util.Vector;
import javax.swing.ComboBoxModel;
import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

public class BasicComboBoxModel implements ComboBoxModel {

	private Vector<BasicComboBoxModelObject> data = new Vector<BasicComboBoxModelObject>();
	private Vector<ListDataListener> list = new Vector<ListDataListener>();
	private BasicComboBoxModelObject selectedItem;

	public BasicComboBoxModel(Integer taCodigos[], String taItems[])
	{
		for (int i = 0; i < taCodigos.length; i++) {
			data.add(new BasicComboBoxModelObject(taCodigos[i], taItems[i]));
		}
	}

	public BasicComboBoxModelObject searchSelectedItem(Integer i)
	{
		for (BasicComboBoxModelObject o : data) {
			if (i.equals(o.getCodigo())) {
				return o;
			}
		}
		return null;
	}

	public BasicComboBoxModelObject searchSelectedItem(String s)
	{
		for (BasicComboBoxModelObject o : data) {
			if (s.equals(o.getDescri())) {
				return o;
			}
		}
		return null;
	}

	public void setSelectedItem(Object anItem)
	{
		selectedItem = anItem instanceof BasicComboBoxModelObject ? (BasicComboBoxModelObject) anItem : null;
		for (ListDataListener l : list) {
			l.contentsChanged(new ListDataEvent(this, javax.swing.event.ListDataEvent.CONTENTS_CHANGED, 0, 0));
		}
	}

	public Object getSelectedItem()
	{
		return selectedItem;
	}

	public int getSize()
	{
		return data.size();
	}

	public Object getElementAt(int index)
	{
		return data.get(index);
	}

	public void addListDataListener(ListDataListener l)
	{
		list.add(l);
	}

	public void removeListDataListener(ListDataListener l)
	{
		list.remove(l);
	}

	public Integer getSelectedCodigo()
	{
		return selectedItem == null ? null : selectedItem.getCodigo();
	}

	public String getSelectedDescri()
	{
		return selectedItem == null ? null : selectedItem.getDescri();
	}
}

class BasicComboBoxModelObject {

	private Integer codigo;
	private String descri;

	public BasicComboBoxModelObject(Integer codigo, String descri)
	{
		this.codigo = codigo;
		this.descri = descri;
	}

	@Override
	public String toString()
	{
		return this.getDescri();
	}

	public Integer getCodigo()
	{
		return codigo;
	}

	public void setCodigo(Integer codigo)
	{
		this.codigo = codigo;
	}

	public String getDescri()
	{
		return descri;
	}

	public void setDescri(String descri)
	{
		this.descri = descri;
	}
}

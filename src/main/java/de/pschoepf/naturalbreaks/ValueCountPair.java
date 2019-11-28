/*
 * Port of Jenks/Fisher breaks originally created in C by Maarten Hilferink.
 * Copyright (C) {2015}  {Philipp Schoepf}
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **/
package de.pschoepf.naturalbreaks;

/**
 * Simple model object to count occurrences per value.
 *
 * @author Philipp Schöpf
 */
public class ValueCountPair {

    // the value
    private double value;
    // occurrence counter
    private int count;

    public ValueCountPair(double value, int count) {
        this.value = value;
        this.count = count;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public void incCount(){
        this.count++;
    }


}
package liac.igmn.util;

public class StringUtil {

	private StringUtil()
	{
	}

	public static String doubleToString(double value, int afterDecimalPoint)
	{
		StringBuffer stringBuffer;
		double temp;
		int dotPosition;
		long precisionValue;

		temp = value * Math.pow(10.0, afterDecimalPoint);
		if (Math.abs(temp) < Long.MAX_VALUE) {
			precisionValue = (temp > 0) ? (long) (temp + 0.5) : -(long) (Math.abs(temp) + 0.5);

			if (precisionValue == 0) {
				stringBuffer = new StringBuffer(String.valueOf(0));
			} else {
				stringBuffer = new StringBuffer(String.valueOf(precisionValue));
			}

			if (afterDecimalPoint == 0) {
				return stringBuffer.toString();
			}

			dotPosition = stringBuffer.length() - afterDecimalPoint;
			while (((precisionValue < 0) && (dotPosition < 1)) || (dotPosition < 0)) {
				if (precisionValue < 0) {
					stringBuffer.insert(1, '0');
				} else {
					stringBuffer.insert(0, '0');
				}

				dotPosition++;
			}
			stringBuffer.insert(dotPosition, '.');
			if ((precisionValue < 0) && (stringBuffer.charAt(1) == '.')) {
				stringBuffer.insert(1, '0');
			} else if (stringBuffer.charAt(0) == '.') {
				stringBuffer.insert(0, '0');
			}
			int currentPos = stringBuffer.length() - 1;
			while ((currentPos > dotPosition) && (stringBuffer.charAt(currentPos) == '0')) {
				stringBuffer.setCharAt(currentPos--, ' ');
			}
			if (stringBuffer.charAt(currentPos) == '.') {
				stringBuffer.setCharAt(currentPos, ' ');
			}

			return stringBuffer.toString().trim();
		}
		return new String("" + value);
	}

	public static String doubleToString(double value, int width, int afterDecimalPoint)
	{
		String tempString = doubleToString(value, afterDecimalPoint);
		char[] result;
		int dotPosition;

		if ((afterDecimalPoint >= width) || (tempString.indexOf('E') != -1)) {
			return tempString;
		}

		result = new char[width];
		for (int i = 0; i < result.length; i++) {
			result[i] = ' ';
		}

		if (afterDecimalPoint > 0) {
			dotPosition = tempString.indexOf('.');
			if (dotPosition == -1) {
				dotPosition = tempString.length();
			} else {
				result[width - afterDecimalPoint - 1] = '.';
			}
		} else {
			dotPosition = tempString.length();
		}

		int offset = width - afterDecimalPoint - dotPosition;
		if (afterDecimalPoint > 0) {
			offset--;
		}

		if (offset < 0) {
			return tempString;
		}

		for (int i = 0; i < dotPosition; i++) {
			result[offset + i] = tempString.charAt(i);
		}

		for (int i = dotPosition + 1; i < tempString.length(); i++) {
			result[offset + i] = tempString.charAt(i);
		}

		return new String(result);
	}
}

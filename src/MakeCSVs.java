import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.awt.image.DataBufferByte;
import javax.imageio.ImageIO;

/*
 * This is a class designed to read a directory of images, convert each one to grayscale, and then write to file the 
 * image name, its class (an integer 0-9), and each individual pixel value of the image. This is so we can use a vector
 * of pixel values as the input parameters for a given machine learning algorithm. This may in fact not be the best approach, 
 * but it seems like the most logical place to start. 
 * 
 * USAGE: First download the distracted drivers dataset from Kaggle (this will require a Kaggle account). Next, unzip only the directory
 * "train". This directory contains 10 sub-directories, one for each class of image. Once "train" has been extracted, call this program with 
 * the path to "train" as the argument. This process took about 35 minutes on the CS machines on campus. It is particularly slow due to ImageIO 
 * and converting to grayscale. This is done to (hopefully) prevent overflowing memory. Once it is complete, you will end up with 10 very large 
 * text files, where each line represents the data for one image. These lines of data can then be used to construct a LabeledPoint RDD in Spark.*/
public class MakeCSVs extends Thread{
	private File rootDir;
	public MakeCSVs(File input){
		this.rootDir = input;
	}
	public static void main(String [] args){
		//args[0] is the root directory containing sub-directories (each sub-directory contains all images of a specific class)		
		File input = new File(args[0]);
		File [] subDirs = input.listFiles();
		for (File f : subDirs){
			Thread t = new Thread(new MakeCSVs(f));
			t.start();
		}
	}
	
	public void run(){
		try {
			this.convertImages();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void convertImages() throws IOException{		
	    File[] fList = this.rootDir.listFiles();
	    String classifier = this.rootDir.getName().substring(1); //directories are labeled as c*, where * is an int from 0-9. We only want the integer value
    	PrintWriter orig = new PrintWriter (new File(this.rootDir.getName()+"_orig"));
	    for (File f : fList){
	    		BufferedImage fullSize = ImageIO.read(f);
	    		toGray(fullSize);
	    		byte [] fullPix = ((DataBufferByte)fullSize.getRaster().getDataBuffer()).getData();
	    		int [] pixels = new int [fullPix.length/3];
	    		for (int i = 0; i < fullPix.length; i+=3)
	    			pixels[i/3] = Byte.toUnsignedInt(fullPix[i]); 
	    		String pixelVals = Arrays.toString(pixels);
	    		orig.write(classifier + "," + f.getName() + "," + pixelVals.substring(1, pixelVals.length()-1).replaceAll(",", "") + "\n");
	    }
	    orig.close();
	}
	
	public static void toGray(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				Color c = new Color(image.getRGB(j, i));
				int red = (int)(c.getRed() * 0.21);
				int green = (int)(c.getGreen() * 0.72);
				int blue = (int)(c.getBlue() *0.07);
				int sum = red + green + blue;
				Color newColor = new Color(sum,sum,sum);
				image.setRGB(j,i,newColor.getRGB());
			}
		}
	}
	
	
	//Likely won't need this, unless we run into OOM issues in which case we can scale images down (although this will likely result 
	// in lost accuracy
	public static BufferedImage createResizedCopy(BufferedImage originalImage, int scaledWidth, int scaledHeight, boolean preserveAlpha){
		System.out.println("resizing...");
		int imageType = preserveAlpha ? BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
		BufferedImage scaledBI = new BufferedImage(scaledWidth, scaledHeight, imageType);
		Graphics2D g = scaledBI.createGraphics();
		if (preserveAlpha)
			g.setComposite(AlphaComposite.Src);
		g.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null);
		g.dispose();
		return scaledBI;
	}
}

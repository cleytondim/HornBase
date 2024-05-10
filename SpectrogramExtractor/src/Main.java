import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.file.Files;

import com.musicg.graphic.GraphicRender;
import com.musicg.wave.Wave;
import com.musicg.wave.WaveHeader;
import com.musicg.wave.extension.Spectrogram;

public class Main {

	public static void main(String[] args) throws Exception {

		File file = new File("Audio.wav");
		InputStream audiofile = new FileInputStream(file);
		GraphicRender grap = new GraphicRender();
		Wave w = new Wave(audiofile);
		Spectrogram spec = new Spectrogram(w);
		grap.renderSpectrogram(spec, "Audio.png");

        System.exit(1);

		// You can do a loop to generate a big ammount of spectrograms from diferents wave files.

	}
        
}

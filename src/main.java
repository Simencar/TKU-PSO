
import java.io.IOException;

public class main {

    public static void main(String[] args) throws IOException {
        TKU_PSO alg = new TKU_PSO();
        alg.run();
        alg.printStats();
    }
}

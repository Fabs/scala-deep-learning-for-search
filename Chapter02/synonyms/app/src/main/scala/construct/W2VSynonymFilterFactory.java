package construct;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.util.TokenFilterFactory;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.util.Map;

public class W2VSynonymFilterFactory extends TokenFilterFactory {
    private final String trainDataFile;
    private final double minAccuracy;
    private final boolean retrain;
    private final Word2Vec model;

    /**
     * Initialize this factory via a set of key-value pairs.
     *
     * @param args
     */
    public W2VSynonymFilterFactory(Map<String, String> args) {
        super(args);
        trainDataFile = args.get("trainDataFile");
        minAccuracy = Double.parseDouble(args.get("minAccuracy"));
        retrain = args.get("retrain").equals("true");
        model = new W2VSynModel(trainDataFile, retrain).model();
        System.out.println("New");
        for(String arg : args.keySet()) {
            System.out.println(arg);
            System.out.println(args.get(arg));
        }
    }

    @Override
    public TokenStream create(TokenStream input) {
        System.out.println("Create");
        return new W2VSynonymFilter(input, model, minAccuracy);
    }
}

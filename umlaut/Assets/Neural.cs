using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(Neural))]
public class NeuralEditor
    : Editor
{
    private bool showLayer0;
    private bool showLayer1;
    private bool showLayer2;

    private bool showWeights10;
    private bool showWeights21;

    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        if (!Application.isPlaying)
            return;

        var self = target as Neural;

        if (GUILayout.Button("Configure 1-1-1"))
            self.Configure(1, 1, 1);
        if (GUILayout.Button("Configure 4-8-6"))
            self.Configure(4, 8, 6);
        if (GUILayout.Button("Configure 8-128-1"))
            self.Configure(8, 128, 1);
        if (GUILayout.Button("Configure 1024-1024-1"))
            self.Configure(1024, 1024, 1);

        if (GUILayout.Button("Reset"))
            self.Reset();
        if (GUILayout.Button("Randomize"))
            self.Randomize();
        if (GUILayout.Button("RandomizeInputs"))
            self.RandomizeInputs();

        EditorGUILayout.Separator();

        showLayer0 = GUILayout.Toggle(showLayer0, "Show Layer 0");
        showLayer1 = GUILayout.Toggle(showLayer1, "Show Layer 1");
        showLayer2 = GUILayout.Toggle(showLayer2, "Show Layer 2");

        showWeights10 = GUILayout.Toggle(showWeights10, "Show Weights 1-0");
        showWeights21 = GUILayout.Toggle(showWeights21, "Show Weights 2-1");

        if (showLayer0)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer0; ++i)
                GUILayout.Label(string.Format("L0: {0:00}: {1,7:F4}", i, self.bias0[i]));
        }

        if (showLayer1)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer1; ++i)
                GUILayout.Label(string.Format("L1: {0:00}, {1,7:F4}", i, self.bias1[i]));
        }

        if (showLayer2)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer2; ++i)
                GUILayout.Label(string.Format("L2: {0:00}, {1,7:F4}", i, self.bias2[i]));
        }

        if (showWeights10)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer1; ++i)
            {
                var label = string.Format("L1: {0:00}, {1,7:F4}", i, self.bias1[i]);

                for (int j = 0; j < self.Layer0; ++j)
                    label += string.Format("{0,7:F3},", self.weights10[i * self.Layer0 + j]);

                GUILayout.Label(label);
            }
        }
        if (showWeights21)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer2; ++i)
            {
                var label = string.Format("L2: {0:00}, {1,7:F4}", i, self.bias2[i]);

                for (int j = 0; j < self.Layer1; ++j)
                    label += string.Format("{0,7:F3},", self.weights21[i * self.Layer1 + j]);

                GUILayout.Label(label);
            }
        }

    }
}
#endif

public class Neural
    : MonoBehaviour
{
    // single neuron takes input in, and output act out
    // sigmoid(x) = 1.0f / (1.0f - exp(-x));
    // out = sigmoid(in)
    // deriv(x) = sigmoid(x) * (1.0f - sigmoid(x))
    // input to a neuron is the weighted sum of output from other neurons
    // each neuron also has a bias value; resting state, that adds to the result

    public virtual void OnEnable()
    {
        if (Automatic)
        {
            Reset();
            Randomize();
            RandomizeInputs();
        }
    }

    public virtual void Update()
    {
        if (Automatic)
            Step();
    }

    private static float rand01() { return Random.Range(0.0f, 1.0f); }
    private static float randh() { return Random.Range(-0.5f, 0.5f); }
    private static float sigm(float x) { return x / (1.0f - Mathf.Exp(-x)); }
    private static float deriv(float x) { return sigm(x) * (1.0f - sigm(x)); }
    private static float tanh(float x) { return (float)System.Math.Tanh((float)x); }

    [System.NonSerialized] public bool Automatic;

    [System.NonSerialized] public int Layer0 = 4;
    [System.NonSerialized] public int Layer1 = 8;
    [System.NonSerialized] public int Layer2 = 6;

    [System.NonSerialized] public float[] bias0;
    [System.NonSerialized] public float[] bias1;
    [System.NonSerialized] public float[] bias2;

    [System.NonSerialized] public float[] state0;
    [System.NonSerialized] public float[] state1;
    [System.NonSerialized] public float[] state2;

    [System.NonSerialized] public float[] weights10;
    [System.NonSerialized] public float[] weights21;

    public void Configure(int layer0, int layer1, int layer2)
    {
        //Debug.LogFormat("Neural.Configure(): {0}:{1}:{2}", layer0, layer1, layer2);

        Layer0 = layer0;
        Layer1 = layer1;
        Layer2 = layer2;

        Reset();
    }

    public void Reset()
    {
        //Debug.LogFormat("Neural.Reset(): {0}:{1}:{2}", Layer0, Layer1, Layer2);

        bias0 = new float[Layer0];
        bias1 = new float[Layer1];
        bias2 = new float[Layer2];

        state0 = new float[Layer0];
        state1 = new float[Layer1];
        state2 = new float[Layer2];

        // [h0: w0..n], [h1: w0..wn]
        weights10 = new float[Layer1 * Layer0];
        weights21 = new float[Layer2 * Layer1];
    }

    public void Randomize()
    {
        //Debug.LogFormat("Neural.Randomize()");

        // randomize biases
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = randh();
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = randh();
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = randh();

        // randomize weights
        for (int i = 0; i < Layer1; ++i)
            for (int j = 0; j < Layer0; ++j)
                weights10[i * Layer0 + j] = randh();

        // randomize weights
        for (int i = 0; i < Layer2; ++i)
            for (int j = 0; j < Layer1; ++j)
                weights21[i * Layer1 + j] = randh();
    }

    public void RandomizeInputs()
    {
        //Debug.LogFormat("Neural.RandomizeInputs()");

        // layer 0 (inputs)
        for (int i = 0; i < Layer0; ++i)
            state0[i] = rand01();
    }

    public void SetInputs(float[] inputs)
    {
        //Debug.LogFormat("Neural.SetInputs()");

        for (int i = 0; i < Layer0; ++i)
            state0[i] = inputs[i];
    }

    public void Step()
    {
        UnityEngine.Profiling.Profiler.BeginSample("Neural.Step");

        //Debug.LogFormat("Neural.Step(): running...");

        // layer 1 (hidden)
        for (int i = 0; i < Layer1; ++i)
        {
            var bias = bias1[i];
            var sum = 0.0f;
            for (int j = 0; j < Layer0; ++j)
                sum += (state0[j] + bias0[j]) * weights10[i * Layer0 + j];
            state1[i] = tanh(bias + sum);
        }

        // layer 2 (output)
        for (int i = 0; i < Layer2; ++i)
        {
            var bias = bias2[i];
            var sum = 0.0f;
            for (int j = 0; j < Layer1; ++j)
                sum += (state1[j] + bias1[j]) * weights21[i * Layer1 + j];
            state2[i] = tanh(bias + sum);
        }

        UnityEngine.Profiling.Profiler.EndSample();
    }

    public void BackProp()
    {
        //Debug.LogFormat("Neural.BackProp(): running...");
    }

    public void LerpTowards(Neural from, float x)
    {
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = Mathf.Lerp(bias0[i], from.bias0[i], x);
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = Mathf.Lerp(bias1[i], from.bias1[i], x);
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = Mathf.Lerp(bias2[i], from.bias2[i], x);

        for (int i = 0; i < Layer0 * Layer1; ++i)
            weights10[i] = Mathf.Lerp(weights10[i], from.weights10[i], x);
        for (int i = 0; i < Layer1 * Layer2; ++i)
            weights21[i] = Mathf.Lerp(weights21[i], from.weights21[i], x);
    }

    public void Mutate(float x)
    {
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = Mathf.Lerp(bias0[i], randh(), x);
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = Mathf.Lerp(bias1[i], randh(), x);
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = Mathf.Lerp(bias2[i], randh(), x);

        for (int i = 0; i < Layer0 * Layer1; ++i)
            weights10[i] = Mathf.Lerp(weights10[i], randh(), x);
        for (int i = 0; i < Layer1 * Layer2; ++i)
            weights21[i] = Mathf.Lerp(weights21[i], randh(), x);
    }
}

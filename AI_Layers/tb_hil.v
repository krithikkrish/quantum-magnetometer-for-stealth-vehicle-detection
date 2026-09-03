`timescale 1ns/1ns

module tb_hil;
    reg  clk;
    reg  rst;
    reg  uart_rx_pin;
    
    wire uart_tx_pin;
    wire mw_gate_pin;
    wire pwm_pin;

    // 50 MHz clock -> 20 ns period
    localparam CLK_PERIOD = 20;
    
    // UART Timing: 115200 baud -> ~8.68 us per bit
    // 8.68 us / 20 ns = 434 clock cycles per bit
    localparam BIT_PERIOD = 434 * CLK_PERIOD; 

    hil_top uut (
        .clk(clk),
        .rst(rst),
        .uart_rx_pin(uart_rx_pin),
        .uart_tx_pin(uart_tx_pin),
        .mw_gate_pin(mw_gate_pin),
        .pwm_pin(pwm_pin)
    );

    // Generate 50 MHz Clock
    always #(CLK_PERIOD/2) clk = ~clk;

    // Task to send a byte over UART to the FPGA (acting as the PC)
    task send_uart_byte;
        input [7:0] data;
        integer i;
        begin
            // Start bit
            uart_rx_pin = 1'b0;
            #(BIT_PERIOD);
            
            // 8 Data bits (LSB first)
            for (i = 0; i < 8; i = i + 1) begin
                uart_rx_pin = data[i];
                #(BIT_PERIOD);
            end
            
            // Stop bit
            uart_rx_pin = 1'b1;
            #(BIT_PERIOD);
        end
    endtask

    initial begin
        $display("==================================================");
        $display("   PHASE 3B: HARDWARE-IN-THE-LOOP TESTBENCH");
        $display("==================================================");
        
        // Initialize
        clk = 0;
        rst = 1;
        uart_rx_pin = 1; // UART idle is high
        
        // Hold reset for 100 ns
        #(CLK_PERIOD * 5);
        rst = 0; 
        $display("[%0t ns] Reset released. Default CPMG sequence should begin.", $time);
        
        // Wait for the default sequence (tau=500) to finish and send SNR
        // A short sequence with tau=500 (10us) and N=4 pulses takes ~40us
        // Plus UART TX time (10 bits * 8.68us = ~87us)
        #(150000); 
        
        $display("[%0t ns] Sending new TAU command from PC -> FPGA...", $time);
        
        // Send a new tau = 520 (0x0208)
        // Send High Byte (0x02)
        send_uart_byte(8'h02);
        #(BIT_PERIOD * 2); // small delay between bytes
        
        // Send Low Byte (0x08)
        send_uart_byte(8'h08);
        $display("[%0t ns] TAU command 520 sent. Waiting for FPGA response...", $time);
        
        // Wait for the new CPMG sequence to run and the new SNR to be transmitted back
        #(150000);
        
        $display("==================================================");
        $display("Simulation complete.");
        $display("Please check ModelSim waveform for:");
        $display("  1. mw_gate_pin (CPMG pi/2 and pi pulses)");
        $display("  2. uart_rx_pin (PC sending tau bytes)");
        $display("  3. uart_tx_pin (FPGA sending SNR result)");
        $display("==================================================");
        $stop;
    end

    // Monitor for debugging in console
    always @(posedge uut.cpmg_done) begin
        $display("[%0t ns] CPMG Sequence Completed in FPGA.", $time);
    end

    always @(posedge uut.snr_ready) begin
        $display("[%0t ns] SNR Calculated: %d", $time, uut.snr_val);
    end

endmodule
